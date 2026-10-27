"""Live screen for the reSpeaker XVF3800 4-mic array (16 kHz 6-channel USB
firmware, channels 3-6 routed to the 4 microphones, see respeaker_route).

Shows where a voice comes from, computed on the PC from the 4 microphones
with SRP-PHAT: a circle with four coloured sides and an arrow, the angle,
a level bar per microphone and a "speaking" light.

Keys: F = the current voice is the front (sets 0 deg)   M = mirror left/right
      O = next microphone order                          Q = quit
"""
import itertools
import subprocess
import threading
import time

import cv2
import numpy as np

FS = 16000
CH = 6
BLOCK = 4096           # 256 ms per direction estimate
C = 343.0
R = 0.032              # assumed microphone circle radius (m)
UP = 16
DEVICE = "hw:CARD=Array,DEV=0"

NAMES = ["FRONT", "LEFT", "BEHIND", "RIGHT"]
COLORS = [(230, 120, 40), (80, 170, 40), (200, 70, 140), (20, 130, 240)]   # BGR: blue, green, purple, orange
# Circular orders of the 4 microphones to try (up to rotation and mirror).
ORDERS = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]]


class Array:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = np.zeros((0, CH), np.int16)
        self.proc = subprocess.Popen(
            ["arecord", "-q", "-D", DEVICE, "-c", str(CH), "-r", str(FS), "-f", "S16_LE", "-t", "raw"],
            stdout=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        frame_bytes = CH * 2
        while True:
            data = self.proc.stdout.read(BLOCK * frame_bytes // 4)
            if not data:
                return
            a = np.frombuffer(data[: len(data) // frame_bytes * frame_bytes], np.int16).reshape(-1, CH)
            with self.lock:
                self.samples = np.concatenate([self.samples, a])[-FS * 2:]

    def latest(self, n):
        with self.lock:
            return self.samples[-n:].astype(np.float64) if len(self.samples) >= n else None


def srp(raw, order):
    """Direction (deg, + = counter-clockwise from mic 0) and peak strength."""
    phis = np.radians(np.array([0, 90, 180, 270])[np.argsort(order)])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    cand = np.radians(np.arange(360))
    n = 1024 * UP
    score = np.zeros(360)
    win = np.hanning(1024)
    for s in range(0, len(raw) - 1024 + 1, 512):
        X = np.fft.rfft(raw[s:s + 1024] * win[:, None], axis=0)
        for (i, j) in pairs:
            G = X[:, i] * np.conj(X[:, j])
            G /= np.abs(G) + 1e-12
            cc = np.fft.irfft(G, n)
            tau = (R * np.cos(cand - phis[j]) - R * np.cos(cand - phis[i])) / C
            score += cc[np.round(tau * FS * UP).astype(int) % n]
    best = int(np.argmax(score))
    return best, score[best] / (np.median(np.abs(score)) + 1e-9)


def main():
    arr = Array()
    offset, mirror, order_i = 0, False, 0
    noise = None
    angle, angle_t, strength = None, 0, 0
    history = []
    cv2.namedWindow("reSpeaker voice direction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("reSpeaker voice direction", 1100, 650)
    while True:
        img = np.full((650, 1100, 3), 250, np.uint8)
        block = arr.latest(BLOCK)
        speaking = False
        levels = [-120] * 4
        if block is not None:
            mics = block[:, 2:6]
            rms = np.sqrt(np.mean(mics ** 2, axis=0))
            levels = list(20 * np.log10(np.maximum(rms, 1) / 32768))
            level = float(np.mean(levels))
            noise = level if noise is None else min(level, noise + 0.02)   # slow-rising noise floor
            noise = min(noise, level)
            speaking = level > noise + 8
            if speaking:
                raw_angle, strength = srp(mics, ORDERS[order_i])
                a = (-raw_angle if mirror else raw_angle) - offset
                angle, angle_t = int(a % 360), time.time()
                history.append(angle)
                history[:] = history[-8:]
        # Circle with four sides (front up, left to the left).
        cx, cy, r = 320, 330, 230
        side = None
        if angle is not None:
            d = (angle + 180) % 360 - 180
            side = 0 if abs(d) <= 45 else 1 if 45 < d < 135 else 3 if -135 < d < -45 else 2
        fresh = angle is not None and time.time() - angle_t < 1.5
        for s, start in enumerate([-135, 135, 45, -45]):   # cv2 angles: 0 = right, clockwise
            col = COLORS[s] if s == side else (225, 225, 225)
            if s == side and not fresh:
                col = tuple(int(c * 0.45 + 255 * 0.55) for c in COLORS[s])
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, col, -1)
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, (255, 255, 255), 6)
        for s, (dx, dy) in enumerate([(0, -1), (-1, 0), (0, 1), (1, 0)]):
            cv2.putText(img, NAMES[s], (cx + int(dx * r * 0.62) - 55, cy + int(dy * r * 0.62) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255) if s == side else COLORS[s], 3)
        cv2.circle(img, (cx, cy), 45, (40, 40, 40), -1)
        cv2.putText(img, "ARRAY", (cx - 38, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if angle is not None:
            t = np.radians(angle)
            end = (int(cx - r * np.sin(t)), int(cy - r * np.cos(t)))   # + = left (counter-clockwise)
            cv2.arrowedLine(img, (cx, cy), end, (30, 30, 30) if fresh else (160, 160, 160), 8, tipLength=0.12)

        # Right side: text and level bars.
        x = 620
        cv2.putText(img, "reSpeaker voice direction", (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        cv2.circle(img, (x + 15, 100), 14, (40, 200, 40) if speaking else (190, 190, 190), -1)
        cv2.putText(img, "SPEAKING" if speaking else "quiet", (x + 40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (40, 160, 40) if speaking else (120, 120, 120), 2)
        txt = "no direction yet" if angle is None else f"{NAMES[side]}   {angle} deg"
        cv2.putText(img, txt, (x, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    COLORS[side] if side is not None else (120, 120, 120), 3)
        cv2.putText(img, f"last: {history}", (x, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
        for i, lv in enumerate(levels):
            y = 250 + i * 45
            frac = min(max((lv + 70) / 70, 0), 1)
            cv2.putText(img, f"mic {i + 1}", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
            cv2.rectangle(img, (x + 80, y), (x + 380, y + 24), (225, 225, 225), -1)
            cv2.rectangle(img, (x + 80, y), (x + 80 + int(300 * frac), y + 24), (220, 140, 60), -1)
            cv2.putText(img, f"{lv:.0f} dB", (x + 390, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        cv2.putText(img, f"front offset {offset} deg, mirror {'on' if mirror else 'off'}, mic order {ORDERS[order_i]}",
                    (x, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        for k, line in enumerate(["F: you are in front and talking -> set front",
                                  "M: mirror left/right    O: next mic order",
                                  "Q: quit"]):
            cv2.putText(img, line, (x, 500 + k * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 110, 200), 2)
        cv2.imshow("reSpeaker voice direction", img)
        key = cv2.waitKey(80) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('f'), ord('F')) and history:
            raw_now = (history[-1] + offset) % 360
            offset = raw_now
        if key in (ord('m'), ord('M')):
            mirror = not mirror
        if key in (ord('o'), ord('O')):
            order_i = (order_i + 1) % len(ORDERS)
    arr.proc.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
