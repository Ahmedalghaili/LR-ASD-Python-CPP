"""Voice gate: is the sound a real voice, and whose voice is it?

The server (ACTIVE_SPEAKER_VOICE_GATE=1) forwards the audio of Zenbo's own
microphone (16 kHz mono int16, as it arrives on port 8897) as UDP to
127.0.0.1:5008. This script runs
  1. Silero VAD: speech or just noise (Zenbo's fan, motors)?
  2. CAM++ (3D-Speaker, zh/en 16 kHz advanced): a voiceprint of the speech,
     compared with the people enrolled in SPEAKERS (cosine score; at least
     VOICE_GATE_THRESHOLD = a known person, else "unknown"),
and sends the result to the server, UDP 127.0.0.1:5009, every 0.2 s while
someone talks: "SPEECH <name> <score>" ("unknown", or "?" before there is
enough speech for a voiceprint), "QUIET" when the speech ends, and "ALIVE"
every second while audio arrives (without it the server turns by direction
alone). The audio may instead come from the reSpeaker on Wi-Fi
(tests/respeaker_wifi_bridge.py), in the same format. The server
then turns Zenbo toward the reSpeaker direction only while a voice is heard.

Enroll a person while the gate runs: write the name to ENROLL_REQUEST
(./run_voice_follow.sh enroll Ahmed does), then the person talks until
ENROLL_SECONDS of speech are collected; the gate saves the voiceprint and
uses it at once. Or, with the gate stopped: python voice_gate.py --enroll Ahmed
List the enrolled people:  python voice_gate.py --list
"""
import argparse
import os
import socket
import time
import wave

import numpy as np
import sherpa_onnx

AUDIO_PORT = 5008
GATE_PORT = 5009
RATE = 16000
HOME = os.path.expanduser("~")
VAD_MODEL = HOME + "/RobotNurseHelper_build/silero-vad/src/silero_vad/data/silero_vad.onnx"
SPEAKER_MODEL = HOME + "/SoloSeniorWatchRobot_build/speaker/campplus_zh_en_advanced.onnx"
SPEAKERS = HOME + "/SoloSeniorWatchRobot_build/speaker/speakers"
ENROLL_REQUEST = HOME + "/SoloSeniorWatchRobot_build/speaker/enroll_request"
# On the robot an enrolled voice scored 0.42-0.82, other people 0.05-0.32.
THRESHOLD = float(os.environ.get("VOICE_GATE_THRESHOLD", "0.38"))
ID_MIN_S = 1.5          # speech needed for a voiceprint
ID_MAX_S = 3.0          # the voiceprint uses at most the last 3 s
ID_EVERY_S = 0.5
REPORT_EVERY_S = 0.2
ENROLL_SECONDS = 10


def make_vad():
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = VAD_MODEL
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.sample_rate = RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


def make_extractor():
    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=SPEAKER_MODEL, num_threads=2))


def voiceprint(extractor, samples):
    """Unit-length CAM++ embedding of float samples."""
    s = extractor.create_stream()
    s.accept_waveform(RATE, samples)
    s.input_finished()
    e = np.array(extractor.compute(s), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def load_speakers():
    """Voiceprints per person: NAME.npy, NAME#2.npy, ... (one per recording)."""
    people = {}
    if os.path.isdir(SPEAKERS):
        for f in sorted(os.listdir(SPEAKERS)):
            if f.endswith(".npy"):
                people.setdefault(f[:-4].split("#")[0], []).append(np.load(os.path.join(SPEAKERS, f)))
    return people


def best_match(people, e):
    """(name, score) of the enrolled person closest to voiceprint e, or None."""
    scores = {name: max(float(p @ e) for p in prints) for name, prints in people.items()}
    return max(scores.items(), key=lambda kv: kv[1], default=None)


def save_person(extractor, person, audio):
    """Saves the mean voiceprint of 2 s pieces of float audio (and the audio)."""
    piece = 2 * RATE
    prints = [voiceprint(extractor, audio[i:i + piece]) for i in range(0, len(audio) - piece + 1, piece)]
    e = np.mean(prints, axis=0)
    e /= np.linalg.norm(e)
    os.makedirs(SPEAKERS, exist_ok=True)
    # A new recording is added next to the earlier ones (NAME#2, NAME#3, ...).
    stem, k = person, 1
    while os.path.exists(os.path.join(SPEAKERS, stem + ".npy")):
        k += 1
        stem = f"{person}#{k}"
    np.save(os.path.join(SPEAKERS, stem + ".npy"), e)
    with wave.open(os.path.join(SPEAKERS, stem + ".wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    agree = min(float(p @ e) for p in prints)
    print(time.strftime("%H:%M:%S"), f"Saved {person} ({len(prints)} pieces; lowest piece score {agree:.2f}).",
          flush=True)


def enroll_request():
    """Name in ENROLL_REQUEST (the file is removed), or None."""
    try:
        with open(ENROLL_REQUEST) as f:
            name = f.read().strip()
        os.remove(ENROLL_REQUEST)
        return name or None
    except OSError:
        return None


def audio_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", AUDIO_PORT))
    sock.settimeout(0.5)
    return sock


def receive(sock):
    """Float samples of one datagram, or None after 0.5 s without audio."""
    try:
        data = sock.recv(65536)
    except socket.timeout:
        return None
    return np.frombuffer(data[:len(data) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768


def run():
    vad, extractor, people = make_vad(), make_extractor(), load_speakers()
    print(f"voice gate: {len(people)} enrolled ({', '.join(people) or 'nobody'}), "
          f"threshold {THRESHOLD}; audio on UDP {AUDIO_PORT}, results to UDP {GATE_PORT}", flush=True)
    sock, out = audio_socket(), socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    speech, speaking = [], False
    name, score, last_id, last_report, last_alive, seen_audio = "?", 0.0, 0.0, 0.0, 0.0, False
    enrolling, enroll_audio, enroll_shown, last_check = None, [], -1, 0.0
    while True:
        if time.time() - last_check >= 1:
            last_check = time.time()
            request = enroll_request()
            if request:
                enrolling, enroll_audio, enroll_shown = request, [], -1
                print(time.strftime("%H:%M:%S"), f"Enrolling {request}: talk normally until "
                      f"{ENROLL_SECONDS} s of speech are heard.", flush=True)
        x = receive(sock)
        if x is None:
            if seen_audio:
                print(time.strftime("%H:%M:%S"), "no audio (server or reSpeaker bridge)", flush=True)
                seen_audio = False
            continue
        if not seen_audio:
            print(time.strftime("%H:%M:%S"), "audio arriving", flush=True)
            seen_audio = True
        vad.accept_waveform(x)
        while not vad.empty():      # finished segments: not needed, keep the queue empty
            vad.pop()
        now = time.time()
        if now - last_alive >= 1:
            last_alive = now
            out.sendto(b"ALIVE", ("127.0.0.1", GATE_PORT))
        if vad.is_speech_detected():
            if not speaking:
                speaking, speech, name, score = True, [], "?", 0.0
            speech.append(x)
            if enrolling:
                enroll_audio.append(x)
                heard = int(sum(len(s) for s in enroll_audio) / RATE)
                if heard != enroll_shown:
                    print(f"  {heard} s of speech", flush=True)
                    enroll_shown = heard
                if heard >= ENROLL_SECONDS:
                    save_person(extractor, enrolling, np.concatenate(enroll_audio))
                    people, enrolling = load_speakers(), None
            n = sum(len(s) for s in speech)
            if n >= ID_MIN_S * RATE and now - last_id >= ID_EVERY_S:
                last_id = now
                e = voiceprint(extractor, np.concatenate(speech)[-int(ID_MAX_S * RATE):])
                best = best_match(people, e)
                score = best[1] if best else 0.0
                new = best[0] if best and score >= THRESHOLD else "unknown"
                if new != name:
                    print(time.strftime("%H:%M:%S"), f"voice: {new} (score {score:.2f})", flush=True)
                name = new
            if now - last_report >= REPORT_EVERY_S:
                last_report = now
                out.sendto(f"SPEECH {name} {score:.2f}".encode(), ("127.0.0.1", GATE_PORT))
        elif speaking:
            speaking = False
            out.sendto(b"QUIET", ("127.0.0.1", GATE_PORT))
            print(time.strftime("%H:%M:%S"), "quiet", flush=True)


def enroll(person):
    vad, extractor = make_vad(), make_extractor()
    sock = audio_socket()
    print(f"Enrolling {person}: talk normally near Zenbo until {ENROLL_SECONDS} s of speech are heard.",
          flush=True)
    speech, last_shown, start = [], -1, time.time()
    while sum(len(s) for s in speech) < ENROLL_SECONDS * RATE:
        if time.time() - start > 180:
            raise SystemExit("not enough speech in 3 minutes; is the server forwarding audio?")
        x = receive(sock)
        if x is None:
            continue
        vad.accept_waveform(x)
        while not vad.empty():
            vad.pop()
        if vad.is_speech_detected():
            speech.append(x)
            heard = int(sum(len(s) for s in speech) / RATE)
            if heard != last_shown:
                print(f"  {heard} s of speech", flush=True)
                last_shown = heard
    save_person(extractor, person, np.concatenate(speech))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enroll", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(", ".join(load_speakers()) or "nobody enrolled")
    elif args.enroll:
        enroll(args.enroll)
    else:
        run()


if __name__ == "__main__":
    main()
