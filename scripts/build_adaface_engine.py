#!/usr/bin/env python
"""AdaFace IR-101 (WebFace12M) → ONNX → TensorRT engine под compute capability 12.0.

  scripts/build_adaface_engine.py \
      --ckpt /data/nimda/qaraqu_face/data/models/adaface_ir101_webface12m.ckpt \
      --code /data/nimda/qaraqu_face/third_party/AdaFace

Вход движка: 1×3×112×112 float32, BGR, (x/255 − 0.5)/0.5 (как в официальном inference.py).
Выход: 512-мерный L2-нормированный вектор.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_onnx(ckpt: Path, code: Path, onnx_path: Path) -> None:
    import torch

    sys.path.insert(0, str(code))
    import net  # type: ignore  # noqa: E402

    model = net.build_model("ir_101")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
    model.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            feat, _norm = self.m(x)
            return feat

    wrap = Wrap(model).eval()
    x = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        wrap,
        x,
        str(onnx_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("onnx →", onnx_path, f"{onnx_path.stat().st_size/1e6:.0f} MB")


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT ≥10: explicit batch по умолчанию
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # TensorRT 11: точность задаётся типами в ONNX (strongly typed); float32-граф + TF32 по умолчанию.
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("TensorRT build failed")
    engine_path.write_bytes(bytes(blob))
    print("engine →", engine_path, f"{engine_path.stat().st_size/1e6:.0f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--code", required=True, help="папка с net.py из репозитория AdaFace")
    ap.add_argument("--onnx", default=str(ROOT / "data/models/adaface_ir101_webface12m.onnx"))
    ap.add_argument("--engine", default=str(ROOT / "data/models/adaface_ir101_webface12m_cc12.engine"))
    args = ap.parse_args()
    onnx_path, engine_path = Path(args.onnx), Path(args.engine)
    if not onnx_path.is_file():
        export_onnx(Path(args.ckpt), Path(args.code), onnx_path)
    build_engine(onnx_path, engine_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
