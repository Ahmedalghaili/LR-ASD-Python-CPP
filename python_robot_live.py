"""Run the original Python LR-ASD pipeline on the live robot streams.

The robot sends protobuf-wrapped JPEG frames on 8895 and raw 16 kHz PCM16
audio on 8897. Ports 8896 and 8898 are accepted as compatibility sockets so
the robot's SocketManager completes its four-socket connection sequence.
"""

import argparse
import collections
import socket
import struct
import threading
import time

import cv2
import numpy as np
import python_speech_features
import torch

from ASD import ASD
from model.faceDetector.s3fd import S3FD
from RobotCommand_pb2 import RobotToServerMessage


HEAD = b"BeginOfADataFrame"
TAIL = b"EndOfADataFrame"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def face_crop(frame, box):
    x1, y1, x2, y2 = box
    size = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = size * 0.55
    left, top = int(cx - half), int(cy - half)
    right, bottom = int(cx + half), int(cy + half)
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - frame.shape[1]), max(0, bottom - frame.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=110)
        left, right, top, bottom = left + pad_l, right + pad_l, top + pad_t, bottom + pad_t
    crop = frame[top:bottom, left:right]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (112, 112))


class RobotStreams:
    def __init__(self, host, base_port):
        self.host = host
        self.base_port = base_port
        self.stop = threading.Event()
        self.frames = collections.deque(maxlen=2)
        self.audio = collections.deque(maxlen=16000 * 10)
        self.frame_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.frame_count = 0
        self.audio_samples = 0
        self.servers = []

    def start(self):
        for port in range(self.base_port, self.base_port + 4):
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, port))
            server.listen(1)
            self.servers.append(server)
            threading.Thread(target=self._accept_loop, args=(server, port), daemon=True).start()
        print(f"Python robot server listening on {self.host or '0.0.0.0'}:{self.base_port}-{self.base_port + 3}", flush=True)

    def _accept_loop(self, server, port):
        while not self.stop.is_set():
            try:
                client, address = server.accept()
                print(f"robot connected to {port}: {address[0]}", flush=True)
                if port == self.base_port:
                    target = self._read_images
                elif port == self.base_port + 2:
                    target = self._read_audio
                else:
                    target = self._read_compat
                threading.Thread(target=target, args=(client,), daemon=True).start()
            except OSError:
                break

    def _read_images(self, client):
        pending = bytearray()
        try:
            while not self.stop.is_set():
                data = client.recv(256 * 1024)
                if not data:
                    break
                pending.extend(data)
                while True:
                    begin = pending.find(HEAD)
                    if begin < 0:
                        if len(pending) > len(HEAD):
                            del pending[:-len(HEAD)]
                        break
                    if begin:
                        del pending[:begin]
                    if len(pending) < len(HEAD) + 4:
                        break
                    length = struct.unpack_from("<i", pending, len(HEAD))[0]
                    if length <= 0 or length > 20 * 1024 * 1024:
                        del pending[:len(HEAD)]
                        continue
                    total = len(HEAD) + 4 + length + len(TAIL)
                    if len(pending) < total:
                        break
                    if pending[len(HEAD) + 4 + length:total] != TAIL:
                        del pending[:len(HEAD)]
                        continue
                    payload = bytes(pending[len(HEAD) + 4:len(HEAD) + 4 + length])
                    del pending[:total]
                    message = RobotToServerMessage()
                    try:
                        message.ParseFromString(payload)
                        image = cv2.imdecode(np.frombuffer(message.jpegdata, np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            with self.frame_lock:
                                self.frames.append(image)
                                self.frame_count += 1
                    except Exception as exc:
                        print(f"image protobuf/decode error: {exc}", flush=True)
        finally:
            client.close()

    def _read_audio(self, client):
        try:
            while not self.stop.is_set():
                data = client.recv(64 * 1024)
                if not data:
                    break
                usable = len(data) - (len(data) % 2)
                samples = np.frombuffer(data[:usable], dtype="<i2").tolist()
                with self.audio_lock:
                    self.audio.extend(samples)
                    self.audio_samples += len(samples)
        finally:
            client.close()

    @staticmethod
    def _read_compat(client):
        try:
            while client.recv(4096):
                pass
        except OSError:
            pass
        finally:
            client.close()

    def get_frame(self):
        with self.frame_lock:
            return self.frames.pop() if self.frames else None

    def get_audio(self):
        with self.audio_lock:
            return np.asarray(self.audio, dtype=np.int16)

    def close(self):
        self.stop.set()
        for server in self.servers:
            server.close()


def main():
    parser = argparse.ArgumentParser(description="Original Python LR-ASD on live robot streams")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--checkpoint", default="weight/pretrain_AVA.model")
    parser.add_argument("--output", default="python_robot_live_output.mp4")
    parser.add_argument("--detector-every", type=int, default=5,
                        help="run original S3FD every N frames and track between runs")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    streams = RobotStreams(args.host, args.port)
    streams.start()
    print("Waiting for the robot app. Press q in the video window to stop.", flush=True)

    detector = S3FD(device="cuda")
    asd = ASD()
    asd.loadParameters(args.checkpoint)
    asd.eval()

    tracks = {}
    next_id = 0
    frame_index = 0
    writer = None
    started = None
    last_report = started

    try:
        while not streams.stop.is_set():
            frame = streams.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))

            if started is None:
                started = time.monotonic()
                last_report = started

            assignments = list(tracks)
            if frame_index % max(1, args.detector_every) == 0 or not tracks:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(rgb, conf_th=0.9, scales=[0.25])
                boxes = [tuple(map(float, det[:4])) for det in detections]
                unmatched = set(tracks)
                assignments = []
                for box in boxes:
                    score, tid = max(((iou(box, tracks[x]["box"]), x) for x in unmatched), default=(0, None))
                    if score < 0.3:
                        tid = next_id
                        next_id += 1
                        tracks[tid] = {"box": box, "faces": collections.deque(maxlen=25), "score": 0.0, "missed": 0}
                    else:
                        unmatched.remove(tid)
                    tracks[tid]["box"] = box
                    tracks[tid]["missed"] = 0
                    assignments.append(tid)

                for tid in unmatched:
                    tracks[tid]["missed"] += 1
            for tid in assignments:
                tracks[tid]["faces"].append(face_crop(frame, tracks[tid]["box"]))

            for tid in list(tracks):
                if tid not in assignments:
                    tracks[tid]["missed"] += 1
                    if tracks[tid]["missed"] > 10:
                        del tracks[tid]

            if frame_index % 5 == 0:
                audio = streams.get_audio()
                if len(audio) >= 1600:
                    mfcc = python_speech_features.mfcc(audio, 16000, numcep=13,
                                                       winlen=0.025, winstep=0.010)
                    audio_end = len(mfcc)
                    with torch.inference_mode():
                        for tid in assignments:
                            faces = list(tracks[tid]["faces"])
                            count = min(len(faces), audio_end // 4)
                            if count < 5:
                                continue
                            video_tensor = torch.tensor(np.asarray(faces[-count:]), dtype=torch.float32,
                                                        device="cuda").unsqueeze(0)
                            audio_tensor = torch.tensor(mfcc[audio_end - count * 4:audio_end],
                                                        dtype=torch.float32, device="cuda").unsqueeze(0)
                            audio_embed = asd.model.forward_audio_frontend(audio_tensor)
                            video_embed = asd.model.forward_visual_frontend(video_tensor)
                            output = asd.model.forward_audio_visual_backend(audio_embed, video_embed)
                            tracks[tid]["score"] = float(asd.lossAV.forward(output, labels=None)[-1])

            for tid in assignments:
                x1, y1, x2, y2 = map(int, tracks[tid]["box"])
                score = tracks[tid]["score"]
                color = (0, 255, 0) if score >= 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{score:.1f}", (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            writer.write(frame)
            if not args.no_window:
                preview = frame
                if frame.shape[1] > 960:
                    preview = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
                cv2.imshow("Original Python LR-ASD - Robot", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                print(f"frames={frame_index} live_fps={frame_index / max(now - started, 1e-6):.1f} "
                      f"robot_frames={streams.frame_count} audio_samples={streams.audio_samples}", flush=True)
                last_report = now
    finally:
        streams.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
