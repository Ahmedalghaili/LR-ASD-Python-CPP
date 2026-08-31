import argparse
import collections
import os
import shutil
import subprocess
import time

import av
import cv2
import numpy as np
import python_speech_features
import torch

from ASD import ASD
from model.faceDetector.s3fd import S3FD
from realtime_video import face_crop, iou


def main():
    parser = argparse.ArgumentParser(description="Live LR-ASD inference from an SRT camera/microphone stream")
    parser.add_argument("--url", default="srt://0.0.0.0:9000?mode=listener&latency=200000")
    parser.add_argument("--checkpoint", default="weight/pretrain_AVA.model")
    parser.add_argument("--window", type=int, default=25, help="rolling video window in frames")
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--output", help="optional silent MP4 recording")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    print(f"Waiting for SRT stream on {args.url}", flush=True)
    bridge = None
    bridge_log = None
    input_url = args.url
    if args.url.startswith("srt://"):
        ffmpeg = "/usr/bin/ffmpeg" if os.path.isfile("/usr/bin/ffmpeg") else shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit("FFmpeg is required for SRT input")
        input_url = "udp://127.0.0.1:19000?fifo_size=1000000&overrun_nonfatal=1"
        bridge_log = open("live_srt_bridge.log", "w")
        bridge = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel", "error",
                "-i", args.url,
                "-map", "0:v:0",
                "-map", "0:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-g", "25",
                "-c:a", "aac",
                "-b:a", "96k",
                "-ar", "16000",
                "-ac", "1",
                "-f", "mpegts",
                "udp://127.0.0.1:19000?pkt_size=1316",
            ],
            stdout=subprocess.DEVNULL,
            stderr=bridge_log,
        )
    try:
        container = av.open(input_url, mode="r")
    except BaseException:
        if bridge is not None:
            bridge.terminate()
            bridge.wait(timeout=5)
        if bridge_log is not None:
            bridge_log.close()
        raise
    video_stream = next(stream for stream in container.streams if stream.type == "video")
    audio_stream = next(stream for stream in container.streams if stream.type == "audio")
    video_stream.thread_type = "AUTO"
    audio_resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    detector = S3FD(device="cuda")
    asd = ASD()
    asd.loadParameters(args.checkpoint)
    asd.eval()

    audio_samples = collections.deque(maxlen=16000 * 3)
    tracks = {}
    next_id = 0
    frame_index = 0
    writer = None
    player = None
    started = time.monotonic()

    try:
        for packet in container.demux(video_stream, audio_stream):
            if packet.stream.type == "audio":
                for decoded in packet.decode():
                    resampled = audio_resampler.resample(decoded)
                    if not isinstance(resampled, list):
                        resampled = [resampled]
                    for audio_frame in resampled:
                        samples = audio_frame.to_ndarray().reshape(-1)
                        audio_samples.extend(samples.tolist())
                continue

            for decoded in packet.decode():
                frame = decoded.to_ndarray(format="bgr24")
                if frame_index < 3:
                    print(f"decoded frame {frame_index}: {frame.shape}", flush=True)
                if writer is None and args.output:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))
                if player is None and not args.no_display:
                    height, width = frame.shape[:2]
                    player = subprocess.Popen(
                        [
                            "/usr/bin/ffplay",
                            "-loglevel", "error",
                            "-fflags", "nobuffer",
                            "-flags", "low_delay",
                            "-f", "rawvideo",
                            "-pixel_format", "bgr24",
                            "-video_size", f"{width}x{height}",
                            "-framerate", "25",
                            "-i", "pipe:0",
                        ],
                        stdin=subprocess.PIPE,
                        bufsize=0,
                    )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(rgb, conf_th=0.9, scales=[0.25])
                if frame_index < 3:
                    print(f"frame {frame_index}: {len(detections)} faces", flush=True)
                boxes = [tuple(map(float, detection[:4])) for detection in detections]
                unmatched = set(tracks)
                assignments = []

                for box in boxes:
                    candidates = [(iou(box, tracks[track_id]["box"]), track_id) for track_id in unmatched]
                    overlap, track_id = max(candidates, default=(0, None))
                    if overlap < 0.3:
                        track_id = next_id
                        next_id += 1
                        tracks[track_id] = {
                            "box": box,
                            "faces": collections.deque(maxlen=args.window),
                            "score": 0.0,
                            "missed": 0,
                        }
                    else:
                        unmatched.remove(track_id)
                    tracks[track_id]["box"] = box
                    tracks[track_id]["faces"].append(face_crop(frame, box))
                    tracks[track_id]["missed"] = 0
                    assignments.append(track_id)

                for track_id in list(tracks):
                    if track_id not in assignments:
                        tracks[track_id]["missed"] += 1
                        if tracks[track_id]["missed"] > 10:
                            del tracks[track_id]

                if frame_index % args.infer_every == 0 and len(audio_samples) >= 4000:
                    samples = np.asarray(audio_samples, dtype=np.int16)
                    mfcc = python_speech_features.mfcc(samples, 16000, numcep=13, winlen=0.025, winstep=0.010)
                    with torch.inference_mode():
                        for track_id in assignments:
                            faces = list(tracks[track_id]["faces"])
                            count = min(len(faces), len(mfcc) // 4)
                            if count < 5:
                                continue
                            video_tensor = torch.tensor(
                                np.asarray(faces[-count:]), dtype=torch.float32, device="cuda"
                            ).unsqueeze(0)
                            audio_tensor = torch.tensor(
                                mfcc[-count * 4:], dtype=torch.float32, device="cuda"
                            ).unsqueeze(0)
                            audio_embed = asd.model.forward_audio_frontend(audio_tensor)
                            video_embed = asd.model.forward_visual_frontend(video_tensor)
                            output = asd.model.forward_audio_visual_backend(audio_embed, video_embed)
                            tracks[track_id]["score"] = float(asd.lossAV.forward(output, labels=None)[-1])

                for track_id in assignments:
                    x1, y1, x2, y2 = map(int, tracks[track_id]["box"])
                    score = tracks[track_id]["score"]
                    color = (0, 255, 0) if score >= 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, f"{score:.1f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                if writer is not None:
                    writer.write(frame)
                if player is not None:
                    try:
                        player.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        return

                frame_index += 1
                if frame_index % 25 == 0:
                    elapsed = time.monotonic() - started
                    print(f"live: {frame_index / elapsed:.1f} fps, {len(assignments)} faces", flush=True)
    finally:
        container.close()
        if bridge is not None:
            bridge.terminate()
            bridge.wait(timeout=5)
        if bridge_log is not None:
            bridge_log.close()
        if writer is not None:
            writer.release()
        if player is not None:
            if player.stdin is not None:
                player.stdin.close()
            player.terminate()
            player.wait(timeout=5)


if __name__ == "__main__":
    main()
