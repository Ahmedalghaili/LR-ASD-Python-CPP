import argparse
import collections
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from ASD import ASD
from model.faceDetector.s3fd import S3FD


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
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=110)
        left, right, top, bottom = left + pad_l, right + pad_l, top + pad_t, bottom + pad_t
    crop = frame[top:bottom, left:right]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (112, 112))


def main():
    parser = argparse.ArgumentParser(description="Near-real-time LR-ASD video inference")
    parser.add_argument("input")
    parser.add_argument("--output", default="realtime_output.mp4")
    parser.add_argument("--checkpoint", default="weight/pretrain_AVA.model")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--window", type=int, default=25, help="rolling video frames")
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--no-pace", action="store_true", help="run as fast as possible")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    with tempfile.TemporaryDirectory(prefix="lr_asd_rt_") as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", args.input, "-ac", "1", "-vn", "-ar", "16000", wav_path,
            "-loglevel", "error",
        ], check=True)
        sr, audio = wavfile.read(wav_path)
        mfcc = python_speech_features.mfcc(audio, sr, numcep=13, winlen=0.025, winstep=0.010)

    detector = S3FD(device="cuda")
    asd = ASD()
    asd.loadParameters(args.checkpoint)
    asd.eval()

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    tracks = {}
    next_id = 0
    frame_index = 0
    source_index = -1
    started = time.monotonic()

    while True:
        wanted_source = round(frame_index * source_fps / args.fps)
        frame = None
        while source_index < wanted_source:
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
            source_index += 1
        if frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb, conf_th=0.9, scales=[0.25])
        boxes = [tuple(map(float, det[:4])) for det in detections]
        unmatched = set(tracks)
        assignments = []
        for box in boxes:
            candidates = [(iou(box, tracks[tid]["box"]), tid) for tid in unmatched]
            score, tid = max(candidates, default=(0, None))
            if score < 0.3:
                tid = next_id
                next_id += 1
                tracks[tid] = {"box": box, "faces": collections.deque(maxlen=args.window), "score": 0.0, "missed": 0}
            else:
                unmatched.remove(tid)
            tracks[tid]["box"] = box
            tracks[tid]["faces"].append(face_crop(frame, box))
            tracks[tid]["missed"] = 0
            assignments.append(tid)

        for tid in list(tracks):
            if tid not in assignments:
                tracks[tid]["missed"] += 1
                if tracks[tid]["missed"] > 10:
                    del tracks[tid]

        if frame_index % args.infer_every == 0:
            audio_end = min(len(mfcc), (frame_index + 1) * 4)
            with torch.inference_mode():
                for tid in assignments:
                    faces = list(tracks[tid]["faces"])
                    count = min(len(faces), audio_end // 4)
                    if count < 5:
                        continue
                    video_tensor = torch.tensor(np.asarray(faces[-count:]), dtype=torch.float32, device="cuda").unsqueeze(0)
                    audio_tensor = torch.tensor(mfcc[audio_end - count * 4:audio_end], dtype=torch.float32, device="cuda").unsqueeze(0)
                    audio_embed = asd.model.forward_audio_frontend(audio_tensor)
                    video_embed = asd.model.forward_visual_frontend(video_tensor)
                    output = asd.model.forward_audio_visual_backend(audio_embed, video_embed)
                    tracks[tid]["score"] = float(asd.lossAV.forward(output, labels=None)[-1])

        for tid in assignments:
            x1, y1, x2, y2 = map(int, tracks[tid]["box"])
            score = tracks[tid]["score"]
            color = (0, 255, 0) if score >= 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, f"{score:.1f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        writer.write(frame)
        frame_index += 1

        if not args.no_pace:
            delay = frame_index / args.fps - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

        if frame_index % int(args.fps) == 0:
            elapsed = time.monotonic() - started
            print(f"processed {frame_index / args.fps:.0f}s, speed {frame_index / elapsed:.1f} fps", flush=True)

    cap.release()
    writer.release()
    elapsed = time.monotonic() - started
    print(f"done: {frame_index} frames in {elapsed:.2f}s ({frame_index / elapsed:.1f} fps) -> {args.output}")


if __name__ == "__main__":
    main()
