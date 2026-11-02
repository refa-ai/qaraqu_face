"""YuNet на ONNX Runtime (CUDA) с тем же интерфейсом, что cv2.FaceDetectorYN.

Декодирование повторяет схему YuNet (якоря stride 8/16/32, score = sqrt(cls * obj), NMS).
Формат строки результата совпадает с cv2: [x, y, w, h, 10 ориентиров, score].
Вход letterbox 640 — сохраняет пропорции кадра, поля серые.
"""

from __future__ import annotations

import cv2
import numpy as np

INPUT = 640
STRIDES = (8, 16, 32)


class YuNetGPU:
    def __init__(self, session, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
        self.sess = session
        self.input_name = session.get_inputs()[0].name
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self._priors = {s: self._make_priors(s) for s in STRIDES}

    def setInputSize(self, size) -> None:  # noqa: N802 — совместимость с cv2.FaceDetectorYN
        pass

    def setScoreThreshold(self, value: float) -> None:  # noqa: N802
        self.score_threshold = float(value)

    @staticmethod
    def _make_priors(stride: int) -> np.ndarray:
        n = INPUT // stride
        ys, xs = np.mgrid[0:n, 0:n]
        cx = (xs.reshape(-1) * stride).astype(np.float32)
        cy = (ys.reshape(-1) * stride).astype(np.float32)
        return np.stack([cx, cy], axis=1)

    def detect(self, frame: np.ndarray):
        return self.run(frame, self.score_threshold)

    def run(self, frame: np.ndarray, score_threshold: float):
        """Без изменения состояния объекта: безопасно из нескольких потоков без общего лока."""
        h, w = frame.shape[:2]
        scale = INPUT / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        sx = sy = 1.0 / scale
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.input_name: blob})
        names = [o.name for o in self.sess.get_outputs()]
        out = dict(zip(names, outs))

        boxes_all, scores_all, kps_all = [], [], []
        for s in STRIDES:
            cls = out[f"cls_{s}"][0, :, 0]
            obj = out[f"obj_{s}"][0, :, 0]
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            bb = out[f"bbox_{s}"][0]
            kp = out[f"kps_{s}"][0]
            pr = self._priors[s]
            cxy = pr + bb[:, :2] * s
            wh = np.exp(bb[:, 2:4]) * s
            x1y1 = cxy - wh / 2
            kps = pr[:, None, :] + kp.reshape(-1, 5, 2) * s
            boxes_all.append(np.concatenate([x1y1, wh], axis=1))
            scores_all.append(score)
            kps_all.append(kps.reshape(-1, 10))

        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)
        kpts = np.concatenate(kps_all)

        keep = scores >= score_threshold
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
        if len(scores) == 0:
            return 1, None
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               float(score_threshold), self.nms_threshold, top_k=self.top_k)
        if len(idx) == 0:
            return 1, None
        idx = np.array(idx).reshape(-1)[: self.top_k]

        res = np.zeros((len(idx), 15), np.float32)
        b = boxes[idx]
        k = kpts[idx]
        res[:, 0] = b[:, 0] * sx
        res[:, 1] = b[:, 1] * sy
        res[:, 2] = b[:, 2] * sx
        res[:, 3] = b[:, 3] * sy
        res[:, 4:14] = k * np.tile([sx, sy], 5)
        res[:, 14] = scores[idx]
        return 1, res


def create_yunet_gpu(onnx_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
    """YuNet на CUDA. Возвращает None, если CUDA-провайдер недоступен — тогда остаётся cv2 на CPU."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in sess.get_providers():
        return None
    return YuNetGPU(sess, score_threshold, nms_threshold, top_k)
