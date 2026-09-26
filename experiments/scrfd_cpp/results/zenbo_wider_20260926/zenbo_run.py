import cv2, glob, json, numpy as np
from detectors import S3FD, SCRFD
a, b = S3FD(), SCRFD()
fs = sorted(f for f in glob.glob('/home/ahmed/Downloads/raw_images/*.jpg') if 'face' not in f and 'outFrame' not in f)
out = []
for f in fs:
    im = cv2.imread(f)
    ra, rb = a.detect(im), b.detect(im)
    out.append({'file': f, 's3fd': ra.tolist(), 'scrfd': rb.tolist()})
json.dump(out, open('zenbo_dets.json', 'w'))
na = np.array([len(o['s3fd']) for o in out]); nb = np.array([len(o['scrfd']) for o in out])
print('frames', len(out))
print('S3FD faces', na.sum(), 'frames with >=1 face', (na > 0).sum())
print('SCRFD faces', nb.sum(), 'frames with >=1 face', (nb > 0).sum())
print('count distribution s3fd', np.bincount(na), 'scrfd', np.bincount(nb))
print('frames with different counts', (na != nb).sum())
