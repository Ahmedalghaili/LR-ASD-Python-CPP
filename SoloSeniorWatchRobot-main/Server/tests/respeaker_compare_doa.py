"""Which reSpeaker XVF3800 direction is more accurate? A guided test.

Compares the chip's readings while you talk from 8 known directions:
  DOA_VALUE         the simple direction we use now (also drives the LEDs)
  PROCESSED         AUDIO_MGR_SELECTED_AZIMUTHS[0]: "processed DoA", picks the
                    speaker's direction from the fixed beams by speech energy
                    (NaN when there is no speech)
  AUTO_BEAM         AUDIO_MGR_SELECTED_AZIMUTHS[1]: the auto-select beam
  FREE_BEAM         AEC_AZIMUTH_VALUES[2]: the free-running (scanning) beam
Read-only: nothing on the board is changed.

The board can be on this PC's USB, or on Zenbo: then the Zenbo app sends the
readings ("respeaker_usb raw ...") and this script follows the server's log
(SERVER_LOG, default below); run the server with
ACTIVE_SPEAKER_FOLLOW=simulation ACTIVE_SPEAKER_DOA=respeaker so Zenbo stays
still.

Stand ~1.5 m away. FRONT (Zenbo's front) is the reference; then go round to
Zenbo's left. At each place press Enter and talk normally for 6 s.
Results: per reading, the error (degrees) at each direction and how much it
jumps; raw samples are saved to respeaker_compare_doa.json.
"""
import json
import math
import os
import re
import statistics
import struct
import time

import usb.core
import usb.util

PLACES = [("FRONT", 0), ("FRONT-LEFT", 45), ("LEFT", 90), ("BACK-LEFT", 135),
          ("BACK", 180), ("BACK-RIGHT", 225), ("RIGHT", 270), ("FRONT-RIGHT", 315)]
TALK_S = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "respeaker_compare_doa.json")
SERVER_LOG = os.environ.get("SERVER_LOG", os.path.expanduser(
    "~/SoloSeniorWatchRobot_build/server_compare_doa.log"))
SOURCES = ["DOA_VALUE", "PROCESSED", "AUTO_BEAM", "FREE_BEAM"]
RAW = re.compile(r"respeaker_usb raw doa=(\d+) processed=(-?\d+) auto=(-?\d+) free=(-?\d+)")


class UsbBoard:
    """The board on this PC's USB."""
    def __init__(self, dev):
        self.dev = dev

    def read(self, resid, cmd, nbytes):
        """Payload of a vendor read (status byte stripped), retrying while busy."""
        t = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
        for _ in range(100):
            r = bytes(self.dev.ctrl_transfer(t, 0, 0x80 | cmd, resid, nbytes + 1, 1000))
            if r[0] == 0:           # status byte: 0 = ok, 64 = retry
                return r[1:]
            time.sleep(0.002)
        raise RuntimeError(f"no answer for resid {resid} cmd {cmd}")

    def collect(self, seconds):
        floats = lambda b: [x[0] for x in struct.iter_unpack("<f", b)]
        deg = lambda r: None if math.isnan(r) else math.degrees(r) % 360
        samples, end = [], time.time() + seconds
        while time.time() < end:
            doa = self.read(20, 18, 4)
            if doa[2]:              # speech
                sel = floats(self.read(35, 11, 8))
                beams = floats(self.read(33, 75, 16))
                samples.append({"DOA_VALUE": doa[0] | doa[1] << 8, "PROCESSED": deg(sel[0]),
                                "AUTO_BEAM": deg(sel[1]), "FREE_BEAM": deg(beams[2])})
            time.sleep(0.1)
        return samples


class ZenboLog:
    """The board on Zenbo: readings from the server's log."""
    def __init__(self, path):
        self.f = open(path, errors="replace")

    def collect(self, seconds):
        self.f.seek(0, os.SEEK_END)
        samples, end = [], time.time() + seconds
        while time.time() < end:
            line = self.f.readline()
            if not line:
                time.sleep(0.05)
                continue
            m = RAW.search(line)
            if m:
                v = [int(x) for x in m.groups()]
                samples.append({s: (None if x < 0 else x) for s, x in zip(SOURCES, v)})
        return samples


def diff(a, b):
    """a - b in degrees, -180..180."""
    return (a - b + 180) % 360 - 180


def circ_median(vals):
    """Median of angles, taken around their circular mean."""
    m = math.degrees(math.atan2(sum(math.sin(math.radians(v)) for v in vals),
                                sum(math.cos(math.radians(v)) for v in vals)))
    return (m + statistics.median(diff(v, m) for v in vals)) % 360


def report(runs):
    """Per source: the median at each place, measured from FRONT."""
    print("\nError in degrees at each place (0 = perfect), and jump = typical")
    print("distance of single readings from that place's median.\n")
    print(f"{'':12}" + "".join(f"{s:>20}" for s in SOURCES))
    table = {s: {} for s in SOURCES}
    for s in SOURCES:
        meds = {}
        for name, _ in PLACES:
            vals = [x[s] for x in runs.get(name, []) if x[s] is not None]
            if len(vals) >= 5:
                med = circ_median(vals)
                meds[name] = (med, statistics.median(abs(diff(v, med)) for v in vals))
        if "FRONT" not in meds:
            continue
        front = meds["FRONT"][0]
        # The reading may count clockwise or counter-clockwise: keep the better fit.
        fits = []
        for sign in (1, -1):
            errs = {n: diff(sign * diff(meds[n][0], front), want)
                    for n, want in PLACES if n in meds}
            fits.append((sum(abs(e) for e in errs.values()), errs))
        errs = min(fits, key=lambda f: f[0])[1]
        for name, _ in PLACES:
            if name in meds:
                table[s][name] = (errs[name], meds[name][1])
    for name, _ in PLACES:
        row = ""
        for s in SOURCES:
            e = table[s].get(name)
            row += f"{'-':>20}" if e is None else f"{e[0]:+8.0f} jump {e[1]:4.0f}  "
        print(f"{name:12}{row}")
    print(f"{'AVERAGE':12}" + "".join(
        f"{'-':>20}" if not table[s] else
        f"{statistics.mean(abs(e) for e, _ in table[s].values()):8.0f} jump "
        f"{statistics.mean(j for _, j in table[s].values()):4.0f}  " for s in SOURCES))

    # Where Zenbo would turn with the server's settings (FREE_BEAM, front,
    # mirror), without fitting anything: + = Zenbo's left.
    try:
        with open(os.path.expanduser("~/SoloSeniorWatchRobot_build/respeaker_seeed_front.json")) as f:
            front = int(json.load(f)["front_deg"])
    except (OSError, ValueError, KeyError):
        front = 90
    front = int(os.environ.get("RESPEAKER_FRONT", front))
    mirror = os.environ.get("RESPEAKER_MIRROR", "1") == "1"
    print(f"\nWith Zenbo's settings (front = array {front}, mirror {'on' if mirror else 'off'}):")
    print(f"{'':12}{'true':>8}{'Zenbo hears':>13}{'error':>8}")
    for name, want in PLACES:
        vals = [x["FREE_BEAM"] for x in runs.get(name, []) if x["FREE_BEAM"] is not None]
        if len(vals) < 5:
            print(f"{name:12}{diff(want, 0):+8.0f}{'-':>13}")
            continue
        doa = diff(circ_median(vals), front)
        if mirror:
            doa = -doa
        side = "front" if abs(doa) <= 25 else "back" if abs(doa) >= 155 else "left" if doa > 0 else "right"
        print(f"{name:12}{diff(want, 0):+8.0f}{doa:+8.0f} {side:<5}{diff(doa, want):+7.0f}")


def main():
    dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
    if dev is not None:
        board = UsbBoard(dev)
        print("reSpeaker on this PC's USB")
    elif os.path.exists(SERVER_LOG):
        board = ZenboLog(SERVER_LOG)
        print("reSpeaker on Zenbo: reading", SERVER_LOG)
    else:
        raise SystemExit(f"reSpeaker not on this PC and no server log at {SERVER_LOG}")
    print(__doc__)
    runs = {}
    for name, _ in PLACES:
        input(f"\nGo to {name}, then press Enter and talk for {TALK_S} s ...")
        runs[name] = board.collect(TALK_S)
        n = len(runs[name])
        print(f"  {n} readings with speech" + ("  (too few: talk louder, or check Zenbo)" if n < 10 else ""))
    with open(OUT, "w") as f:
        json.dump(runs, f)
    report(runs)
    print(f"\nRaw samples saved to {OUT}")


if __name__ == "__main__":
    main()
