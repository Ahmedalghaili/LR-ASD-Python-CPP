"""Python ports of the two production detectors in LRASDWorker.cpp /
SCRFDDetector.hpp. Preprocessing, decoding and NMS follow the C++ exactly;
only the score threshold is exposed so a full PR curve can be computed."""
import numpy as np
import cv2
import onnxruntime as ort

S3FD_PATH = "/home/ahmed/SoloSeniorWatchRobot_build/lr-asd/s3fd_270x480.onnx"
SCRFD_PATH = ("/home/ahmed/Desktop/Ahmed/experiments/LR-ASD-main/experiments/"
              "scrfd_cpp/models/scrfd_2.5g_bnkps.dynamic.onnx")



def _greedy(boxes, thr, plus1):
    if len(boxes) == 0:
        return boxes.reshape(-1, 5)
    boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")]
    e = 1.0 if plus1 else 0.0
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + e) * (y2 - y1 + e)
    keep, alive = [], np.ones(len(boxes), bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        iw = np.maximum(0, np.minimum(x2[i], x2) - np.maximum(x1[i], x1) + e)
        ih = np.maximum(0, np.minimum(y2[i], y2) - np.maximum(y1[i], y1) + e)
        inter = iw * ih
        u = area[i] + area - inter
        ov = np.where(u > 0, inter / np.where(u > 0, u, 1), 0)
        alive &= ~(ov > thr)
    return boxes[keep]


def nms(boxes, thr):
    return _greedy(boxes, thr, True)


def nms_plain(boxes, thr):
    return _greedy(boxes, thr, False)


class S3FD:
    DEPLOYED_THRESHOLD = 0.9

    def __init__(self, providers=("CPUExecutionProvider",)):
        o = ort.SessionOptions()
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(S3FD_PATH, o, providers=list(providers))
        steps, mins = [4, 8, 16, 32, 64, 128], [16, 32, 64, 128, 256, 512]
        h, w = [270 // 4], [480 // 4]
        for z in range(1, 6):
            if z == 3:
                h.append(h[-1] // 2); w.append(w[-1] // 2)
            else:
                h.append((h[-1] + 1) // 2); w.append((w[-1] + 1) // 2)
        pri = []
        for z in range(6):
            ys, xs = np.meshgrid(np.arange(h[z]), np.arange(w[z]), indexing="ij")
            cx = (xs.ravel() + .5) * steps[z] / 480
            cy = (ys.ravel() + .5) * steps[z] / 270
            pri.append(np.stack([cx, cy, np.full_like(cx, mins[z] / 480),
                                 np.full_like(cx, mins[z] / 270)], 1))
        self.priors = np.concatenate(pri).astype(np.float32)

    def detect(self, frame, threshold=DEPLOYED_THRESHOLD):
        H, W = frame.shape[:2]
        r = cv2.resize(frame, (480, 270)).astype(np.float32)
        r -= np.array([123, 117, 104], np.float32)
        x = r.transpose(2, 0, 1)[None]
        loc, conf = self.s.run(["loc", "conf"], {"image": x})
        loc, conf = loc[0], conf[0][:, 1]
        m = conf >= threshold
        p, l = self.priors[m], loc[m]
        cx = p[:, 0] + l[:, 0] * .1 * p[:, 2]
        cy = p[:, 1] + l[:, 1] * .1 * p[:, 3]
        w = p[:, 2] * np.exp(l[:, 2] * .2)
        h = p[:, 3] * np.exp(l[:, 3] * .2)
        b = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H, conf[m]], 1)
        # C++ NMS for S3FD uses its own iou() helper with threshold 0.1.
        return nms_plain(b, 0.1)




class SCRFD:
    DEPLOYED_THRESHOLD = 0.5
    W, H = 480, 288

    def __init__(self, providers=("CPUExecutionProvider",)):
        o = ort.SessionOptions()
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(SCRFD_PATH, o, providers=list(providers))
        self.inp = self.s.get_inputs()[0].name
        self.outs = [o.name for o in self.s.get_outputs()][:6]
        self.centers = []
        for level in range(3):
            st = 8 << level
            cols, rows = self.W // st, self.H // st
            idx = np.arange(rows * cols * 2)
            self.centers.append(np.stack([(idx // 2) % cols * st,
                                          (idx // 2) // cols * st], 1).astype(np.float32))

    def detect(self, frame, threshold=DEPLOYED_THRESHOLD):
        ratio = frame.shape[0] / frame.shape[1]
        if ratio > self.H / self.W:
            rh = self.H; rw = int(rh / ratio)
        else:
            rw = self.W; rh = int(rw * ratio)
        rw, rh = max(1, rw), max(1, rh)
        scale = rh / frame.shape[0]
        pad = np.zeros((self.H, self.W, 3), np.uint8)
        pad[:rh, :rw] = cv2.resize(frame, (rw, rh))
        x = ((pad[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.s.run(self.outs, {self.inp: np.ascontiguousarray(x)})
        res = []
        for level in range(3):
            st = 8 << level
            sc = outs[level].reshape(-1)
            d = outs[level + 3].reshape(-1, 4)
            m = sc >= threshold
            c = self.centers[level][m]
            dd = d[m] * st
            b = np.stack([(c[:, 0] - dd[:, 0]) / scale, (c[:, 1] - dd[:, 1]) / scale,
                          (c[:, 0] + dd[:, 2]) / scale, (c[:, 1] + dd[:, 3]) / scale, sc[m]], 1)
            res.append(b)
        b = np.concatenate(res)
        b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
        return nms(b, 0.4)
