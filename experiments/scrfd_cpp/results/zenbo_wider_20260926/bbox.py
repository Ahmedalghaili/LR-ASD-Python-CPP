# NumPy equivalent of widerface_evaluate/box_overlaps.pyx (+1 pixel convention).
import numpy as np
def bbox_overlaps(boxes, query):
    b = boxes.astype(np.float64)[:, None, :]; q = query.astype(np.float64)[None, :, :]
    iw = np.minimum(b[..., 2], q[..., 2]) - np.maximum(b[..., 0], q[..., 0]) + 1
    ih = np.minimum(b[..., 3], q[..., 3]) - np.maximum(b[..., 1], q[..., 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    qa = (q[..., 2] - q[..., 0] + 1) * (q[..., 3] - q[..., 1] + 1)
    ba = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)
    ov = inter / (ba + qa - inter)
    return np.where((iw > 0) & (ih > 0), ov, 0.0)
