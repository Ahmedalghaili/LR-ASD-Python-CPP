"""Send the reSpeaker XVF3800's built-in voice direction to the server
(ACTIVE_SPEAKER_DOA=respeaker), so Zenbo turns toward the voice.

Reads DOA_VALUE (resid 20, cmd 18: angle + speech flag) and, while the array
hears speech, sends the direction as UDP text to 127.0.0.1:5006: degrees,
+ = Zenbo's left, 0 = Zenbo's front, -180..180.

Two sources:
  USB (default): the array plugged into this PC (USB firmware).
  RESPEAKER_SOURCE=wifi: the XIAO ESP32S3 on the array (respeaker_wifi_doa.ino,
  I2S firmware) broadcasts "DOA <angle> <speech>" to UDP port 5007.

Front is the one set with F in respeaker_seeed_screen.py (stand in front of
Zenbo, talk, press F); without it the XIAO side of the board is the front.
With Wi-Fi the screen cannot read the array, so set the front with
RESPEAKER_FRONT=<the array angle printed while talking from Zenbo's front>.
The array's angle slides for a moment when a new voice starts (seen:
168, -152, -98, -44, -20 within 2 s), so a direction is sent only once the
last STEADY readings agree within STEADY_DEG.
If Zenbo turns the wrong way (left/right swapped), run with RESPEAKER_MIRROR=1.
"""
import json
import os
import socket
import time

import usb.core
import usb.util

from respeaker_seeed_screen import CALIBRATION, read_doa

PORT = 5006
WIFI_PORT = 5007
STEADY = 4          # readings, 0.1 s apart
STEADY_DEG = 15


class UsbSource:
    """The array on this PC's USB; reconnects after unplugging."""
    def __init__(self):
        self.dev = None

    def read(self):
        if self.dev is None:
            self.dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
            if self.dev is None:
                time.sleep(1)
                return None
            print(time.strftime("%H:%M:%S"), "reSpeaker array found", flush=True)
        try:
            d = read_doa(self.dev)
        except usb.core.USBError as e:
            print(time.strftime("%H:%M:%S"), "reSpeaker array lost:", e, flush=True)
            self.dev = None
            return None
        time.sleep(0.1)
        return d


class WifiSource:
    """Readings broadcast by the XIAO on the array."""
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", WIFI_PORT))
        self.sock.settimeout(2)
        self.seen = None

    def read(self):
        try:
            data, addr = self.sock.recvfrom(64)
        except socket.timeout:
            if self.seen:
                print(time.strftime("%H:%M:%S"), "no readings from the XIAO", flush=True)
                self.seen = None
            return None
        if addr[0] != self.seen:
            print(time.strftime("%H:%M:%S"), "readings from the XIAO at", addr[0], flush=True)
            self.seen = addr[0]
        parts = data.decode(errors="replace").split()
        if len(parts) != 3 or parts[0] != "DOA":
            return None
        return int(parts[1]), parts[2] == "1"


def main():
    wifi = os.environ.get("RESPEAKER_SOURCE") == "wifi"
    source = WifiSource() if wifi else UsbSource()
    try:
        with open(CALIBRATION) as f:
            front = int(json.load(f)["front_deg"])
    except (OSError, ValueError, KeyError):
        front = 90                  # XIAO side of the board
    if os.environ.get("RESPEAKER_FRONT"):
        front = int(os.environ["RESPEAKER_FRONT"])
    print(f"source {'Wi-Fi' if wifi else 'USB'}; front = array angle {front} deg; "
          f"sending to 127.0.0.1:{PORT}", flush=True)
    mirror = os.environ.get("RESPEAKER_MIRROR") == "1"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last = None
    recent = []
    while True:
        d = source.read()
        if d is None or not d[1]:
            recent.clear()
            continue
        doa = (d[0] - front + 180) % 360 - 180
        if mirror:
            doa = -doa
        recent = (recent + [doa])[-STEADY:]
        if len(recent) < STEADY or any(abs((x - doa + 180) % 360 - 180) > STEADY_DEG for x in recent):
            continue
        sock.sendto(str(doa).encode(), ("127.0.0.1", PORT))
        if doa != last:
            print(time.strftime("%H:%M:%S"), f"voice at {doa} deg (+ = left), array angle {d[0]}", flush=True)
            last = doa


if __name__ == "__main__":
    main()
