"""Инференс ONNX-графов на PyTorch (CUDA).

Те же сети, что раньше жили в TensorRT: ArcFace w600k_r50 и AdaFace IR-101.
Веса читаются из ONNX и считаются torch-операторами, чтобы векторы картотеки
совпадали с уже посчитанными.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:  # noqa: BLE001
    pass

# Одна видеокарта на YOLO и оба эмбеддера. Замок не даёт пикам памяти наложиться.
_GPU_LOCK = threading.RLock()


def gpu_lock() -> threading.RLock:
    return _GPU_LOCK


def _pads_nchw(pads: list[int]) -> tuple[int, int, int, int]:
    """ONNX pads [h0, w0, h1, w1] → F.pad (left, right, top, bottom)."""
    if len(pads) != 4:
        return (0, 0, 0, 0)
    h0, w0, h1, w1 = (int(v) for v in pads)
    return (w0, w1, h0, h1)


class TorchRunner:
    """Один ONNX-граф, исполненный на PyTorch. Вход и выход — float32 NCHW."""

    def __init__(self, path: str | Path):
        import onnx
        import torch
        from onnx import numpy_helper

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"модель не найдена: {self.path}")
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = onnx.load(str(self.path))
        self.input_names = [i.name for i in model.graph.input]
        self.output_names = [o.name for o in model.graph.output]
        init_names = {i.name for i in model.graph.initializer}
        # вход графа, который на самом деле инициализатор, в input_names не нужен
        self.input_names = [n for n in self.input_names if n not in init_names]

        self._const: dict[str, torch.Tensor] = {}
        for init in model.graph.initializer:
            arr = np.array(numpy_helper.to_array(init), copy=True)
            ten = torch.from_numpy(np.ascontiguousarray(arr))
            if ten.dtype.is_floating_point:
                ten = ten.to(dtype=torch.float32, device=self.device)
            self._const[init.name] = ten

        self._nodes: list[tuple] = []
        for node in model.graph.node:
            attrs: dict = {}
            for a in node.attribute:
                attrs[a.name] = _read_attr(a, numpy_helper, torch, self.device)
            if node.op_type == "Constant" and node.output:
                val = attrs.get("value")
                if isinstance(val, torch.Tensor):
                    self._const[node.output[0]] = val
                continue
            self._nodes.append((node.op_type, list(node.input), list(node.output), attrs))
        self._lock = threading.Lock()

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self._torch
        env: dict[str, torch.Tensor] = dict(self._const)
        for name, arr in feeds.items():
            ten = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            env[name] = ten.to(device=self.device, non_blocking=False)
        with self._lock:
            with _GPU_LOCK:
                with torch.inference_mode():
                    for op, inputs, outputs, attrs in self._nodes:
                        args = [env[n] for n in inputs if n]
                        y = _apply(op, args, attrs)
                        if len(outputs) == 1:
                            env[outputs[0]] = y
                        else:
                            for name, part in zip(outputs, y):
                                env[name] = part
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        out: dict[str, np.ndarray] = {}
        for name in self.output_names:
            out[name] = env[name].detach().float().cpu().numpy()
        return out


def _read_attr(a, numpy_helper, torch, device):
    if a.type == 1:  # FLOAT
        return float(a.f)
    if a.type == 2:  # INT
        return int(a.i)
    if a.type == 3:  # STRING
        return a.s
    if a.type == 4:  # TENSOR
        arr = np.array(numpy_helper.to_array(a.t), copy=True)
        ten = torch.from_numpy(np.ascontiguousarray(arr))
        if ten.dtype.is_floating_point:
            ten = ten.to(dtype=torch.float32, device=device)
        return ten
    if a.type == 6:  # FLOATS
        return [float(v) for v in a.floats]
    if a.type == 7:  # INTS
        return [int(v) for v in a.ints]
    return None


def _apply(op: str, args: list, attrs: dict):
    import torch
    import torch.nn.functional as F

    if op == "Conv":
        x, w = args[0], args[1]
        b = args[2] if len(args) > 2 else None
        if b is not None:
            b = b.reshape(-1)
        pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        stride = tuple(int(v) for v in attrs.get("strides", [1, 1]))
        dil = tuple(int(v) for v in attrs.get("dilations", [1, 1]))
        groups = int(attrs.get("group", 1))
        h0, w0, h1, w1 = (pads + [0, 0, 0, 0])[:4]
        if h0 == h1 and w0 == w1:
            return F.conv2d(x, w, b, stride, (h0, w0), dil, groups)
        x = F.pad(x, _pads_nchw(pads))
        return F.conv2d(x, w, b, stride, 0, dil, groups)

    if op == "PRelu":
        x, slope = args[0], args[1]
        if x.ndim == 4 and slope.ndim == 3:
            slope = slope.reshape(1, slope.shape[0], 1, 1)
        elif x.ndim == 4 and slope.ndim == 1 and slope.numel() > 1:
            slope = slope.reshape(1, -1, 1, 1)
        return torch.where(x >= 0, x, slope * x)

    if op == "BatchNormalization":
        x, scale, bias, mean, var = args[:5]
        eps = float(attrs.get("epsilon", 1e-5))
        return F.batch_norm(
            x,
            mean.reshape(-1),
            var.reshape(-1),
            scale.reshape(-1),
            bias.reshape(-1),
            training=False,
            eps=eps,
        )

    if op == "Add":
        return args[0] + args[1]

    if op == "MaxPool":
        x = args[0]
        k = tuple(int(v) for v in attrs.get("kernel_shape", [1, 1]))
        stride = tuple(int(v) for v in attrs.get("strides", list(k)))
        dil = tuple(int(v) for v in attrs.get("dilations", [1, 1]))
        pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        if any(pads):
            x = F.pad(x, _pads_nchw(pads))
        return F.max_pool2d(x, k, stride, 0, dil, ceil_mode=bool(attrs.get("ceil_mode", 0)))

    if op == "Flatten":
        axis = int(attrs.get("axis", 1))
        return torch.flatten(args[0], start_dim=axis)

    if op == "Gemm":
        a, b = args[0], args[1]
        if int(attrs.get("transA", 0)):
            a = a.transpose(-1, -2)
        if int(attrs.get("transB", 0)):
            b = b.transpose(-1, -2)
        y = float(attrs.get("alpha", 1.0)) * (a @ b)
        if len(args) > 2:
            y = y + float(attrs.get("beta", 1.0)) * args[2]
        return y

    if op == "Reshape":
        shape = [int(v) for v in args[1].detach().cpu().reshape(-1).tolist()]
        src = list(args[0].shape)
        for i, s in enumerate(shape):
            if s == 0 and i < len(src):
                shape[i] = int(src[i])
        # батч: константа графа зашита на 1 кадр — берём фактический размер батча со входа
        if shape and shape[0] == 1 and src and src[0] > 1:
            shape[0] = int(src[0])
        return args[0].reshape(shape)

    if op == "ReduceL2":
        axes = attrs.get("axes")
        if axes is None and len(args) > 1:
            axes = [int(v) for v in args[1].detach().cpu().reshape(-1).tolist()]
        dims = tuple(int(v) for v in axes) if axes else None
        keep = bool(attrs.get("keepdims", 1))
        if dims:
            return torch.linalg.vector_norm(args[0], ord=2, dim=dims, keepdim=keep)
        return torch.linalg.vector_norm(args[0], ord=2, keepdim=keep)

    if op == "Div":
        return args[0] / args[1]

    if op == "Relu":
        return F.relu(args[0])

    if op == "Sigmoid":
        return torch.sigmoid(args[0])

    if op == "Transpose":
        perm = attrs.get("perm")
        if perm:
            return args[0].permute(*[int(v) for v in perm])
        return args[0].transpose(-1, -2)

    raise NotImplementedError(f"ONNX-оператор {op} не поддержан в PyTorch-рантайме")
