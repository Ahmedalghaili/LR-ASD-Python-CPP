"""Simple screen: where the reSpeaker XVF3800 itself says the voice is
(its built-in DOA_VALUE, read over USB). One arrow, one word.

The array's angle is counter-clockwise, seen from the mic side, with mic 1 at
(+33, -33) mm and mic 2 at (+33, +33) mm (AEC_MIC_ARRAY_GEO); so 90 deg is the
side between mic 2 and mic 3 (the XIAO board) and 270 deg is the USB port.
By default FRONT is the XIAO side and BEHIND the USB port.

Keys: F = I am in front and talking (sets front)   Q = quit
"""
import json
import os
import time

import cv2
import numpy as np
import usb.core
import usb.util

CALIBRATION = os.path.expanduser("~/SoloSeniorWatchRobot_build/respeaker_seeed_front.json")
NAMES = ["FRONT", "LEFT", "BEHIND", "RIGHT"]
COLORS = [(230, 120, 40), (80, 170, 40), (200, 70, 140), (20, 130, 240)]   # BGR


def read_doa(dev):
    """(angle, speaking) from DOA_VALUE (resid 20, cmd 18), or None."""
    t = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
    for _ in range(50):
        r = bytes(dev.ctrl_transfer(t, 0, 0x80 | 18, 20, 5, 1000))
        if r[0] == 0:               # status byte: 0 = ok, otherwise retry
            return r[1] | r[2] << 8, bool(r[3])
        time.sleep(0.01)
    return None


def main():
    dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
    try:
        with open(CALIBRATION) as f:
            front = int(json.load(f)["front_deg"])
    except (OSError, ValueError, KeyError):
        front = 90                  # XIAO side of the board
    angle, angle_t, raw = None, 0, None
    cv2.namedWindow("Voice direction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Voice direction", 700, 760)
    while True:
        img = np.full((760, 700, 3), 250, np.uint8)
        speaking = False
        if dev is not None:
            d = read_doa(dev)
            if d is not None and d[1]:
                speaking, raw = True, d[0]
                angle, angle_t = (raw - front) % 360, time.time()
        fresh = angle is not None and time.time() - angle_t < 2
        side = None
        if angle is not None:
            a = (angle + 180) % 360 - 180
            side = 0 if abs(a) <= 45 else 1 if a > 45 and a < 135 else 3 if -135 < a < -45 else 2
        cx, cy, r = 350, 380, 260
        for s, start in enumerate([-135, 135, 45, -45]):
            col = COLORS[s] if s == side and fresh else (225, 225, 225)
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, col, -1)
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, (255, 255, 255), 6)
        for s, (dx, dy) in enumerate([(0, -1), (-1, 0), (0, 1), (1, 0)]):
            cv2.putText(img, NAMES[s], (cx + int(dx * r * 0.62) - 55, cy + int(dy * r * 0.62) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255) if s == side and fresh else COLORS[s], 3)
        if angle is not None:
            t = np.radians(angle)
            end = (int(cx - r * np.sin(t)), int(cy - r * np.cos(t)))   # + = left
            cv2.arrowedLine(img, (cx, cy), end, (30, 30, 30) if fresh else (170, 170, 170), 10, tipLength=0.12)
        if dev is None:
            title, col = "Array not found - plug it in", (40, 40, 200)
        elif angle is None:
            title, col = "Talk to start", (120, 120, 120)
        else:
            title, col = f"Voice: {NAMES[side]}  ({raw} deg)", COLORS[side] if fresh else (150, 150, 150)
        cv2.putText(img, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        cv2.circle(img, (650, 45), 16, (40, 200, 40) if speaking else (200, 200, 200), -1)
        cv2.putText(img, "F = I am in front (set front)    Q = quit", (30, 735),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 110, 200), 2)
        cv2.imshow("Voice direction", img)
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('f'), ord('F')) and raw is not None:
            front = raw
            angle = 0
            os.makedirs(os.path.dirname(CALIBRATION), exist_ok=True)
            with open(CALIBRATION, "w") as f:
                json.dump({"front_deg": front}, f)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
