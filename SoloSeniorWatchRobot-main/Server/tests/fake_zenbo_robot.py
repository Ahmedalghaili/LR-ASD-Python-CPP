"""Fake Zenbo Junior client for testing the server without the robot.

Streams a video file and its audio to the server exactly like the robot app
(JPEG frames in RobotToServerMessage on 8895, raw 16 kHz PCM16 audio on 8897),
receives RobotCommand messages on 8896 and logs them as JSON lines.

Closed loop: the robot camera is simulated as a window cropped from the wider
video. A body rotation command (degree) pans that window at the configured
speed, so the server sees faces shift exactly as after a real turn.

Mock DOA (--doa-from-csv LOG.csv): while the recorded LR-ASD log says someone
is speaking, send an "onVoiceDetect" message every 400 ms like the robot app,
with doa = the speaker's direction relative to the robot body (degrees,
+ = left) plus noise. look_at_user_doa commands turn the body by that amount,
which is how this mock models Zenbo's utility.lookAtUser().

Usage:
  python fake_zenbo_robot.py VIDEO [--host 127.0.0.1] [--seconds N]
         [--log commands.jsonl] [--window 0.75] [--doa-from-csv LOG.csv]
Requires: opencv-python, numpy, protobuf, ffmpeg, protoc.
"""
import argparse
import json
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
PROTO_DIR = os.path.dirname(HERE)
GEN = tempfile.mkdtemp(prefix="fake_zenbo_pb_")
subprocess.run(["protoc", "-I", PROTO_DIR, "--python_out", GEN, "RobotCommand.proto"], check=True)
sys.path.insert(0, GEN)
import RobotCommand_pb2 as pb  # noqa: E402

HEAD, TAIL = b"BeginOfADataFrame", b"EndOfADataFrame"
FPS = 25
HFOV = 62.5            # degrees seen by the simulated robot camera
VFOV = 48.9
CONNECT_PITCH = 30     # the Zenbo app calls moveHead(0, 30) after connecting
BODY_DEG_PER_S = 45.0  # simulated Zenbo rotation speed


def connect(host, port):
    s = socket.create_connection((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


class Robot:
    def __init__(self):
        self.yaw = 0.0            # current body yaw, + = turned left
        self.target_yaw = 0.0
        self.pitch = CONNECT_PITCH
        self.face = None
        self.lock = threading.Lock()

    def step(self, dt):
        with self.lock:
            delta = self.target_yaw - self.yaw
            m = BODY_DEG_PER_S * dt
            self.yaw += max(-m, min(m, delta))


def read_commands(sock, robot, log, t0):
    buf = bytearray()
    while True:
        try:
            data = sock.recv(4096)
        except OSError:
            return
        if not data:
            return
        buf.extend(data)
        while True:
            b = buf.find(HEAD)
            e = buf.find(TAIL, b + len(HEAD)) if b >= 0 else -1
            if b < 0 or e < 0:
                break
            cmd = pb.RobotCommand()
            cmd.ParseFromString(bytes(buf[b + len(HEAD):e]))
            del buf[:e + len(TAIL)]
            entry = {"t": round(time.time() - t0, 2),
                     "fields": [f.name for f, _ in cmd.ListFields()]}
            with robot.lock:
                if cmd.HasField("degree") and not cmd.HasField("x"):
                    robot.target_yaw += cmd.degree
                    entry["rotate"] = cmd.degree
                if cmd.HasField("yaw"):
                    robot.pitch = cmd.pitch
                    entry["head"] = [cmd.yaw, cmd.pitch]
                if cmd.HasField("face"):
                    robot.face = cmd.face
                    entry["face"] = cmd.face
                if cmd.HasField("look_at_user_doa"):
                    robot.target_yaw += cmd.look_at_user_doa
                    entry["look_at_user_doa"] = round(cmd.look_at_user_doa, 1)
                if cmd.HasField("bodyspeed"):
                    entry["bodyspeed"] = cmd.bodyspeed
                entry["body_yaw_target"] = robot.target_yaw
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print("robot received", entry, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--log", default="fake_zenbo_commands.jsonl")
    ap.add_argument("--window", type=float, default=0.75,
                    help="fraction of the video width the robot camera sees")
    ap.add_argument("--size", default="640x360", help="frame size sent to the server")
    ap.add_argument("--doa-from-csv", help="LR-ASD per-frame log used to mock DOA reports")
    ap.add_argument("--doa-noise", type=float, default=8.0, help="DOA noise std (degrees)")
    args = ap.parse_args()
    out_w, out_h = map(int, args.size.split("x"))

    wav = os.path.join(GEN, "audio.raw")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.video, "-ac", "1", "-ar", "16000",
                    "-f", "s16le", wav], check=True)
    audio = np.fromfile(wav, np.int16)

    images = connect(args.host, 8895)
    commands = connect(args.host, 8896)
    sound = connect(args.host, 8897)
    messages = connect(args.host, 8898)  # the robot app opens all four sockets

    robot = Robot()
    t0 = time.time()
    log = open(args.log, "w")
    threading.Thread(target=read_commands, args=(commands, robot, log, t0), daemon=True).start()

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    win_w = int(W * args.window)
    win_h = int(win_w * out_h / out_w)
    px_per_deg = win_w / HFOV
    samples_per_frame = 16000 // FPS
    total = int(args.seconds * FPS) if args.seconds else 10 ** 9

    speech = {}  # 25 fps frame -> (score, box centre x) of the most confident speaker
    if args.doa_from_csv:
        import csv
        for row in csv.DictReader(open(args.doa_from_csv)):
            score = float(row["score"])
            if score < 0:
                continue
            f = int(row["frame"])
            cxp = (float(row["x1"]) + float(row["x2"])) / 2
            if f not in speech or score > speech[f][0]:
                speech[f] = (score, cxp)
    rng = np.random.default_rng(0)
    doa_sent = 0

    frame = None
    src_index = -1
    sent = 0
    next_time = time.time()
    while sent < total:
        want = int(sent / FPS * src_fps)  # resample the source to 25 fps
        while src_index < want:
            ok, f = cap.read()
            if not ok:
                break
            frame, src_index = f, src_index + 1
        if src_index < want or frame is None:
            break
        robot.step(1.0 / FPS)
        with robot.lock:
            yaw, pitch = robot.yaw, robot.pitch
        # Turning left (+yaw) moves the view window left in the world image;
        # tilting the head up (+pitch) moves it up. The video centre is what
        # the camera sees at yaw 0 and the connect pitch.
        cx = W / 2 - yaw * px_per_deg
        cy = H / 2 - (pitch - CONNECT_PITCH) * (win_h / VFOV)
        # Beyond the edge of the video the robot sees an empty (black) room,
        # so every turn changes the view as a real turn would.
        x0 = int(round(cx - win_w / 2))
        y0 = int(round(cy - win_h / 2))
        window = np.zeros((win_h, win_w, 3), np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + win_w, W), min(y0 + win_h, H)
        if sx1 > sx0 and sy1 > sy0:
            window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        view = cv2.resize(window, (out_w, out_h))
        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
        msg = pb.RobotToServerMessage()
        msg.jpegdata = jpg.tobytes()
        msg.jpegdatalength = len(msg.jpegdata)
        msg.event_time.GetCurrentTime()
        payload = msg.SerializeToString()
        images.sendall(HEAD + len(payload).to_bytes(4, "little") + payload + TAIL)
        if speech and sent % 10 == 0 and sent in speech:
            world = (W / 2 - speech[sent][1]) / px_per_deg  # + = left of the start pose
            doa = world - yaw + rng.normal(0, args.doa_noise)
            vmsg = pb.RobotToServerMessage()
            vmsg.event_time.GetCurrentTime()
            vmsg.description = "onVoiceDetect"
            vmsg.doa = int(round(doa))
            vmsg.sound_level = 50
            vmsg.vad_status = "Speech"
            vp = vmsg.SerializeToString()
            messages.sendall(HEAD + len(vp).to_bytes(4, "little") + vp + TAIL)
            doa_sent += 1
        a = audio[sent * samples_per_frame:(sent + 1) * samples_per_frame]
        if len(a):
            sound.sendall(a.tobytes())
        sent += 1
        next_time += 1.0 / FPS
        delay = next_time - time.time()
        if delay > 0:
            time.sleep(delay)

    print(f"sent {sent} frames and {doa_sent} DOA reports in {time.time() - t0:.1f}s; "
          f"final body yaw {robot.yaw:+.1f} deg", flush=True)
    time.sleep(1.0)
    for s in (images, commands, sound, messages):
        s.close()


if __name__ == "__main__":
    main()
