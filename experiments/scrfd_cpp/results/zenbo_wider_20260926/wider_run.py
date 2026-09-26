import os, sys, cv2, numpy as np, time
from detectors import S3FD, SCRFD
name = sys.argv[1]
det = S3FD() if name == 's3fd' else SCRFD()
root = 'wider/WIDER_val/images'; out = 'wider_pred_' + name
t0 = time.time(); n = 0
for ev in sorted(os.listdir(root)):
    os.makedirs(os.path.join(out, ev), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(root, ev))):
        im = cv2.imread(os.path.join(root, ev, f))
        b = det.detect(im, threshold=0.02)
        with open(os.path.join(out, ev, f[:-4] + '.txt'), 'w') as fh:
            fh.write(f[:-4] + '\n%d\n' % len(b))
            for x in b:
                fh.write('%.1f %.1f %.1f %.1f %.4f\n' % (x[0], x[1], x[2] - x[0], x[3] - x[1], x[4]))
        n += 1
print(name, 'images', n, 'seconds', round(time.time() - t0))
