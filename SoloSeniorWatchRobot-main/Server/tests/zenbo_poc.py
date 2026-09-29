"""Zenbo compatibility proof of concept (original ASUS Zenbo, Android 6).

A small stand-alone server that speaks the robot app's protocol, so it can
test the robot WITHOUT the C++ LR-ASD server (stop that server first: both
use ports 8895-8898). It never changes the LR-ASD code.

It measures every movement from the camera itself: the image shift between
the frame before and after a command (phase correlation) gives the real
direction and size of the turn, and the camera field of view.

Steps (safe order):
  observe  camera FPS, audio level and robot messages; no movement
  head     head yaw 0 -> +15 -> +30 -> ... -> -30 -> 0 and pitch 20/40/30
  body     body rotation +15/-15/+30/-30 (only with --allow-body)
  voice    needs the new Zenbo app: counts DOA events in three listening
           configurations (default, DOA monitor, always-listen)
  look     needs the new Zenbo app: turns with utility.lookAtUser(doa) after
           each DOA event; watches the camera for interference
  track    needs the new Zenbo app: runs utility.trackUser() and watches
           the camera for interference

During every step it also reports camera FPS, the longest gap between
frames, and the audio level, so interference with the camera or audio
stream is visible.

Simulation (no robot):
  python zenbo_poc.py head --listen 127.0.0.1 &
  python fake_zenbo_robot.py VIDEO --window 0.5 --host 127.0.0.1

Real robot (the Zenbo app connects to this PC as usual):
  python zenbo_poc.py observe head
  python zenbo_poc.py body --allow-body
Report: zenbo_poc_report.json (and printed).
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = tempfile.mkdtemp(prefix="zenbo_poc_pb_")
subprocess.run(["protoc", "-I", os.path.dirname(HERE), "--python_out", GEN, "RobotCommand.proto"],
               check=True)
sys.path.insert(0, GEN)
import RobotCommand_pb2 as pb  # noqa: E402

HEAD, TAIL = b"BeginOfADataFrame", b"EndOfADataFrame"
PORTS = {"image": 8895, "command": 8896, "audio": 8897, "message": 8898}
ASSUMED_HFOV = 62.5
START_PITCH = 30  # the Zenbo app's pose after connecting


class Link:
    """Accepts the robot app's four connections and collects its streams."""

    def __init__(self, listen, port_offset=0):
        self.lock = threading.Lock()
        self.cmd_sock = None
        self.frames = []          # (t, gray 320-wide frame)
        self.frame_times = []
        self.audio = []           # (t, dBFS of ~100 ms)
        self.audio_bytes = 0
        self.messages = []        # (t, dict)
        self.connected = threading.Event()
        for name, port in PORTS.items():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((listen, port + port_offset))
            srv.listen(1)
            threading.Thread(target=self._accept, args=(name, srv), daemon=True).start()

    def _accept(self, name, srv):
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[link] {name} connected from {addr[0]}", flush=True)
            if name == "command":
                with self.lock:
                    self.cmd_sock = conn
                self.connected.set()
            else:
                target = {"image": self._images, "audio": self._audio,
                          "message": self._messages}[name]
                threading.Thread(target=target, args=(conn,), daemon=True).start()

    @staticmethod
    def _frames(conn):
        """Yield protobuf payloads framed as HEAD + int32 LE length + payload + TAIL."""
        buf = bytearray()
        while True:
            data = conn.recv(65536)
            if not data:
                return
            buf.extend(data)
            while True:
                b = buf.find(HEAD)
                if b < 0 or len(buf) < b + len(HEAD) + 4:
                    break
                n = int.from_bytes(buf[b + len(HEAD):b + len(HEAD) + 4], "little")
                end = b + len(HEAD) + 4 + n
                if len(buf) < end + len(TAIL):
                    break
                payload = bytes(buf[b + len(HEAD) + 4:end])
                del buf[:end + len(TAIL)]
                yield payload

    def _images(self, conn):
        try:
            for payload in self._frames(conn):
                msg = pb.RobotToServerMessage()
                msg.ParseFromString(payload)
                if not msg.jpegdata:
                    continue
                img = cv2.imdecode(np.frombuffer(msg.jpegdata, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
                now = time.time()
                with self.lock:
                    self.frames.append((now, small))
                    self.frames = self.frames[-100:]
                    self.frame_times.append(now)
        except OSError:
            pass

    def _audio(self, conn):
        pending = b""
        try:
            while True:
                data = conn.recv(8192)
                if not data:
                    return
                pending += data
                with self.lock:
                    self.audio_bytes += len(data)
                while len(pending) >= 3200:  # 100 ms of 16 kHz PCM16
                    chunk = np.frombuffer(pending[:3200], np.int16).astype(np.float64)
                    pending = pending[3200:]
                    rms = math.sqrt(float(np.mean(chunk ** 2))) if len(chunk) else 0.0
                    with self.lock:
                        self.audio.append((time.time(), 20 * math.log10(max(rms, 1.0) / 32768)))
        except OSError:
            pass

    def _messages(self, conn):
        try:
            for payload in self._frames(conn):
                msg = pb.RobotToServerMessage()
                msg.ParseFromString(payload)
                entry = {"description": msg.description}
                for field in ("doa", "sound_level", "vad_status"):
                    if msg.HasField(field):
                        entry[field] = getattr(msg, field)
                with self.lock:
                    self.messages.append((time.time(), entry))
                print(f"[robot] {entry}", flush=True)
        except OSError:
            pass

    def send(self, **fields):
        cmd = pb.RobotCommand(**fields)
        with self.lock:
            sock = self.cmd_sock
        if sock is None:
            raise RuntimeError("the robot command socket is not connected")
        sock.sendall(HEAD + cmd.SerializeToString() + TAIL)
        print(f"[send] {fields}", flush=True)

    def latest_frame(self):
        with self.lock:
            return self.frames[-1] if self.frames else (0, None)

    def stats(self, t0, t1):
        """Camera and audio health between t0 and t1."""
        with self.lock:
            times = [t for t in self.frame_times if t0 <= t <= t1]
            levels = [d for t, d in self.audio if t0 <= t <= t1]
            msgs = [m for t, m in self.messages if t0 <= t <= t1]
        span = max(t1 - t0, 1e-6)
        gaps = np.diff(times) if len(times) > 1 else np.array([span])
        return {
            "camera_fps": round(len(times) / span, 1),
            "max_frame_gap_s": round(float(gaps.max()), 2),
            "audio_dbfs_median": round(float(np.median(levels)), 1) if levels else None,
            "audio_dbfs_max": round(float(np.max(levels)), 1) if levels else None,
            "audio_chunks_per_s": round(len(levels) / span, 1),  # 10 = full stream
            "messages": len(msgs),
            "camera_problems": [m["description"] for m in msgs
                                if m["description"].startswith("camera_")],
        }


def image_shift(a, b):
    """Shift (dx, dy) in pixels of the scene from frame a to frame b, and the
    phase-correlation response (low = unreliable)."""
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(np.float32(a), np.float32(b), win)
    return dx, dy, response


def measure(link, command, settle, label):
    """Send a command and measure how the camera view moved."""
    time.sleep(0.3)
    _, before = link.latest_frame()
    t0 = time.time()
    link.send(**command)
    time.sleep(settle)
    t1 = time.time()
    _, after = link.latest_frame()
    result = {"step": label, "command": command, **link.stats(t0, t1)}
    if before is None or after is None:
        result["error"] = "no camera frames"
        return result
    dx, dy, response = image_shift(before, after)
    deg_per_px = ASSUMED_HFOV / before.shape[1]
    result.update({
        "image_dx_px": round(dx, 1), "image_dy_px": round(dy, 1),
        "match_response": round(response, 3),
        # The scene moves right in the image (dx > 0) when the camera turns left.
        "camera_turned": ("left" if dx > 0 else "right") if abs(dx) >= 2 else "no horizontal turn",
        "camera_tilted": ("up" if dy > 0 else "down") if abs(dy) >= 2 else "no tilt",
        "turn_deg_at_assumed_fov": round(abs(dx) * deg_per_px, 1),
        "frame_width_px": before.shape[1],
    })
    print(json.dumps(result), flush=True)
    return result


def step_observe(link, args):
    t0 = time.time()
    time.sleep(args.seconds)
    r = {"step": "observe", **link.stats(t0, time.time())}
    print(json.dumps(r), flush=True)
    return [r]


def step_head(link, args):
    limit = min(abs(args.max_yaw), 45)
    results = [measure(link, {"yaw": 0, "pitch": START_PITCH, "headspeed": 2}, args.settle,
                       "head home")]
    for yaw in (15, 30, 15, 0, -15, -30, -15, 0):
        yaw = max(-limit, min(limit, yaw))
        results.append(measure(link, {"yaw": yaw, "pitch": START_PITCH, "headspeed": 2},
                               args.settle, f"head yaw {yaw:+d}"))
    for pitch in (20, 40, START_PITCH):
        results.append(measure(link, {"yaw": 0, "pitch": pitch, "headspeed": 2}, args.settle,
                               f"head pitch {pitch}"))
    return results


def step_body(link, args):
    if not args.allow_body:
        print("body: skipped (add --allow-body; make sure the robot has free space)")
        return [{"step": "body", "skipped": "no --allow-body"}]
    results = []
    for degree in (15, -15, 30, -30):
        results.append(measure(link, {"degree": degree}, args.settle + abs(degree) / 30,
                               f"body {degree:+d}"))
    return results


def step_voice(link, args):
    results = []
    link.send(poc_action="sdk_version")
    link.send(poc_action="states_on")
    for name, on, off in (("default", None, None),
                          ("doa_monitor", "doa_monitor_on", "doa_monitor_off"),
                          ("always_listen", "always_listen_on", "always_listen_off")):
        if on:
            link.send(poc_action=on)
        print(f"voice/{name}: speak from different sides for {args.seconds:.0f} s", flush=True)
        t0 = time.time()
        time.sleep(args.seconds)
        t1 = time.time()
        if off:
            link.send(poc_action=off)
        with link.lock:
            msgs = [m for t, m in link.messages if t0 <= t <= t1]
        doas = [m["doa"] for m in msgs if "doa" in m]
        r = {"step": f"voice {name}", **link.stats(t0, t1),
             "doa_events": len(doas), "doa_values": doas[:30],
             "poc_results": [m["description"] for m in msgs
                             if m["description"].startswith("poc_result")]}
        print(json.dumps(r), flush=True)
        results.append(r)
    link.send(poc_action="states_off")
    return results


def step_look(link, args):
    link.send(poc_action="states_on")
    link.send(poc_action="doa_monitor_on")
    t0 = time.time()
    turns = 0
    seen = 0
    while time.time() - t0 < args.seconds:
        with link.lock:
            doas = [(t, m["doa"]) for t, m in link.messages if "doa" in m and t > t0]
        if len(doas) > seen:
            seen = len(doas)
            link.send(look_at_user_doa=float(doas[-1][1]))
            turns += 1
            time.sleep(4)
        time.sleep(0.2)
    link.send(poc_action="doa_monitor_off")
    link.send(poc_action="states_off")
    r = {"step": "look", "look_at_user_calls": turns, **link.stats(t0, time.time())}
    print(json.dumps(r), flush=True)
    return [r]


def step_track(link, args):
    link.send(poc_action="states_on")
    t_before = time.time()
    time.sleep(3)
    before = link.stats(t_before, time.time())
    t0 = time.time()
    link.send(poc_action="track_user")
    time.sleep(args.seconds)
    link.send(poc_action="cancel_all")
    during = link.stats(t0, time.time())
    time.sleep(3)
    after = link.stats(time.time() - 3, time.time())
    link.send(poc_action="states_off")
    r = {"step": "track", "before": before, "during": during, "after": after}
    print(json.dumps(r), flush=True)
    return [r]


def summarize(results):
    """Head/body direction and an estimate of the real camera field of view."""
    summary = {}
    for kind, key in (("head yaw", "yaw"), ("body", "degree")):
        rows = [r for r in results if r.get("step", "").startswith(kind) and "image_dx_px" in r
                and r.get("match_response", 0) > 0.05]
        pairs = []
        prev = 0
        for r in rows:
            value = r["command"].get(key, 0)
            change = value - prev if key == "yaw" else value
            prev = value if key == "yaw" else 0
            if change and abs(r["image_dx_px"]) >= 2:
                pairs.append((change, r["image_dx_px"], r["frame_width_px"]))
        if not pairs:
            continue
        # + command with the scene moving right (dx > 0) means + turns the camera left.
        same = sum(1 for c, dx, _ in pairs if (c > 0) == (dx > 0))
        px_per_deg = float(np.median([abs(dx) / abs(c) for c, dx, _ in pairs]))
        summary[kind] = {
            "positive_command_turns_camera": "left" if same >= len(pairs) / 2 else "right",
            "consistent_moves": f"{max(same, len(pairs) - same)}/{len(pairs)}",
            "estimated_hfov_deg": round(pairs[0][2] / px_per_deg, 1) if px_per_deg else None,
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+",
                    choices=["observe", "head", "body", "voice", "look", "track"])
    ap.add_argument("--listen", default="0.0.0.0")
    ap.add_argument("--port-offset", type=int, default=0,
                    help="listen on 8895+N...; for simulation next to a running server")
    ap.add_argument("--seconds", type=float, default=20, help="duration of observe/voice/look/track")
    ap.add_argument("--settle", type=float, default=3.0, help="wait after each movement (s)")
    ap.add_argument("--max-yaw", type=int, default=30, help="head yaw limit (Zenbo: 45)")
    ap.add_argument("--allow-body", action="store_true", help="allow body rotation")
    ap.add_argument("--report", default="zenbo_poc_report.json")
    args = ap.parse_args()

    link = Link(args.listen, args.port_offset)
    print(f"waiting for the robot app on ports 8895-8898 ({args.listen}) ...", flush=True)
    link.connected.wait()
    t_wait = time.time()
    while link.latest_frame()[1] is None and time.time() - t_wait < 20:
        time.sleep(0.2)
    time.sleep(2)

    steps = {"observe": step_observe, "head": step_head, "body": step_body,
             "voice": step_voice, "look": step_look, "track": step_track}
    results = []
    try:
        for name in args.steps:
            print(f"=== {name}", flush=True)
            results += steps[name](link, args)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        try:
            link.send(yaw=0, pitch=START_PITCH, headspeed=2)  # head back home
        except Exception as exc:
            print("could not return the head home:", exc)
    report = {"results": results, "summary": summarize(results)}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print("=== summary\n" + json.dumps(report["summary"], indent=1))
    print(f"report saved in {args.report}")
    time.sleep(1)


if __name__ == "__main__":
    main()
