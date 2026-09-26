import cv2, json, numpy as np
d = json.load(open('zenbo_dets.json'))
def iou(p, q):
    iw = max(0, min(p[2], q[2]) - max(p[0], q[0])); ih = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = iw * ih; return i / ((p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i)
dis = []
for k, o in enumerate(d):
    ua = [x for x in o['s3fd'] if not any(iou(x, y) > .3 for y in o['scrfd'])]
    ub = [y for y in o['scrfd'] if not any(iou(x, y) > .3 for x in o['s3fd'])]
    if ua or ub: dis.append((k, len(ua), len(ub)))
print('disagreement frames', len(dis)); print(dis)
tiles = []
for k, _, _ in dis:
    o = d[k]; im = cv2.imread(o['file'])
    for x in o['s3fd']: cv2.rectangle(im, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0, 0, 255), 3); cv2.putText(im, 'S3FD %.2f' % x[4], (int(x[0]), int(x[3]) + 22), 0, .7, (0, 0, 255), 2)
    for x in o['scrfd']: cv2.rectangle(im, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0, 255, 0), 2); cv2.putText(im, 'SCRFD %.2f' % x[4], (int(x[0]), int(x[1]) - 6), 0, .7, (0, 255, 0), 2)
    cv2.putText(im, '#%d' % k, (10, 40), 0, 1.2, (255, 0, 255), 3)
    tiles.append(cv2.resize(im, (480, 360)))
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
for s in range(0, len(rows), 3):
    cv2.imwrite('sheet_%d.jpg' % (s // 3), np.vstack(rows[s:s+3]))
print('sheets', (len(rows) + 2) // 3)
