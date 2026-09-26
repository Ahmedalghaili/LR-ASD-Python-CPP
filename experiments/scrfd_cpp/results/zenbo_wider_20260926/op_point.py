"""Precision/recall on WIDER FACE val at each detector's deployed threshold.
Matching follows the official evaluation: IoU >= 0.5, greedy by score, and a
detection matched to a face outside the setting (ignored face) is neither a TP
nor an FP. Also breaks recall down by face height in pixels."""
import os, sys, numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps

G = 'wider'
gt = loadmat(os.path.join(G, 'wider_face_val.mat'))
sets = {s: loadmat(os.path.join(G, 'wider_%s_val.mat' % s))['gt_list'] for s in ('easy', 'medium', 'hard')}
THR = {'s3fd': 0.9, 'scrfd': 0.5}
BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10000)]


def load(pred_dir, ev, name):
    with open(os.path.join(pred_dir, ev, name + '.txt')) as f:
        lines = f.read().split('\n')[2:]
    a = np.array([list(map(float, l.split())) for l in lines if l.strip()]).reshape(-1, 5)
    a[:, 2] += a[:, 0]; a[:, 3] += a[:, 1]
    return a


for det in ('s3fd', 'scrfd'):
    thr = THR[det]
    for s, lst in sets.items():
        tp = fp = nface = 0
        bin_tot = np.zeros(len(BINS)); bin_hit = np.zeros(len(BINS))
        for i, ev in enumerate(gt['event_list']):
            ev = str(ev[0][0])
            for j, fn in enumerate(gt['file_list'][i][0]):
                fn = str(fn[0][0])
                g = gt['face_bbx_list'][i][0][j][0].astype(float)
                keep = lst[i][0][j][0].ravel() - 1
                valid = np.zeros(len(g), bool); valid[keep] = True
                nface += valid.sum()
                gb = g.copy(); gb[:, 2] += gb[:, 0]; gb[:, 3] += gb[:, 1]
                p = load('wider_pred_' + det, ev, fn)
                p = p[p[:, 4] >= thr]
                p = p[np.argsort(-p[:, 4])]
                hit = np.zeros(len(g), bool)
                if len(p) and len(g):
                    ov = bbox_overlaps(p[:, :4], gb)
                    for k in range(len(p)):
                        m = ov[k].argmax()
                        if ov[k, m] >= 0.5:
                            if not valid[m]:
                                continue  # matched an ignored face
                            if not hit[m]:
                                hit[m] = True; tp += 1
                            else:
                                fp += 1
                        else:
                            fp += 1
                else:
                    fp += len(p)
                h = g[:, 3]
                for b, (lo, hi) in enumerate(BINS):
                    sel = valid & (h >= lo) & (h < hi)
                    bin_tot[b] += sel.sum(); bin_hit[b] += (sel & hit).sum()
        print('%-5s %-6s thr=%.2f recall=%.1f%% precision=%.1f%% TP=%d FP=%d faces=%d' % (
            det, s, thr, 100 * tp / nface, 100 * tp / max(1, tp + fp), tp, fp, nface))
        if s == 'hard':
            print('      recall by face height (px):', ', '.join(
                '%d-%s: %.1f%% (n=%d)' % (lo, hi if hi < 10000 else '+', 100 * bin_hit[b] / max(1, bin_tot[b]), bin_tot[b])
                for b, (lo, hi) in enumerate(BINS)))
