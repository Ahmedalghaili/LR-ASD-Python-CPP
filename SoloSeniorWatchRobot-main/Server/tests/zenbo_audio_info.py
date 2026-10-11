"""Which of Zenbo's microphones can an app read?

Needs Zenbo Nurse Helper V2 with the audio diagnostics (poc_action
"audio_info" and "mic_probe_devices"). Stop the LR-ASD server first (same
ports), run this, connect V2, and keep talking from the robot's left side
while it records.

1. audio_info: every input device Android lists (type, channel counts),
   the sound cards, and the audio configuration files. Saved in
   audio_info_out/ (one text file per item).
2. mic_probe_devices: 1 s recordings from every input device directly
   (setPreferredDevice) with 2, 4 and 8 channels, 16 and 48 kHz. For each,
   the tool reports how many channels carry sound, and how many of those
   are different microphones (not copies).
"""
import os
import re
import sys
import time

import numpy as np

import zenbo_poc as poc
import zenbo_mic_probe as mp

OUT = "audio_info_out"


class InfoLink(mp.ProbeLink):
    def __init__(self, *a, **k):
        self.info = []
        super().__init__(*a, **k)

    def _messages(self, conn):
        try:
            for payload in self._frames(conn):
                msg = poc.pb.RobotToServerMessage()
                msg.ParseFromString(payload)
                d = msg.description
                if d.startswith("audio_info"):
                    self.info.append(d)
                    print("[robot] " + d.split("\n", 1)[0], flush=True)
                elif d.startswith(("mic_probe", "mic_record")):
                    audio = None
                    if msg.audio_probe and msg.audio_probe_channels > 0:
                        a = np.frombuffer(msg.audio_probe, np.int16)
                        ch = msg.audio_probe_channels
                        audio = a[: len(a) // ch * ch].reshape(-1, ch)
                    with self.rec_lock:
                        self.recordings.append((d, audio, msg.audio_probe_rate))
                    print("[robot] " + d, flush=True)
                elif d.startswith(("poc_result", "update_check")):
                    print("[robot] " + d, flush=True)
                with self.lock:
                    self.messages.append((time.time(), {"description": d}))
        except OSError:
            pass


def wait_text(link, text, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if any(text in i for i in link.info):
            return True
        time.sleep(0.5)
    return False


def summarize(audio, rate):
    levels = [mp.dbfs(audio[:, i]) for i in range(audio.shape[1])]
    live = [i for i, l in enumerate(levels) if l > -85]
    distinct = []
    for k, i in enumerate(live):
        for j in live[k + 1:]:
            x, y = audio[:, i].astype(float), audio[:, j].astype(float)
            if np.array_equal(audio[:, i], audio[:, j]):
                continue
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 1.0
            if corr < 0.999:
                us, cons = mp.gcc_phat(x, y, rate)
                distinct.append(f"{i}-{j} corr={corr:.3f} delay={us:.0f}us")
    return levels, live, distinct


def main():
    os.makedirs(OUT, exist_ok=True)
    link = InfoLink("0.0.0.0")
    print("waiting for Zenbo Nurse Helper V2 on port 8895 ...", flush=True)
    link.connected.wait()
    time.sleep(4)

    link.send(poc_action="audio_info")
    if not wait_text(link, "audio_info done", 90):
        print("audio_info did not finish (old app version?)", flush=True)
    for k, text in enumerate(link.info):
        head, _, body = text.partition("\n")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", head.replace("audio_info", "").strip()) or "item"
        with open(os.path.join(OUT, f"{k:02d}_{name}.txt"), "w") as f:
            f.write(head + "\n" + body)
    devices = next((t for t in link.info if t.startswith("audio_info devices")), "")
    print("\n=== input devices\n" + devices.partition("\n")[2], flush=True)

    if "--info-only" in sys.argv:
        print(f"saved in {OUT}/", flush=True)
        return
    print("=== recording from every device: KEEP TALKING from the robot's left side", flush=True)
    link.send(speak_sentence="Microphone test. Please keep talking on my left side.")
    time.sleep(5)
    before = len(link.recordings)
    link.send(poc_action="mic_probe_devices")
    mp.wait_for(link, "mic_probe_done", 1, 400)
    link.send(speak_sentence="Thank you.")

    print("\n=== results: channels with sound / different microphones", flush=True)
    best = (0, "")
    for d, audio, rate in link.recordings[before:]:
        if audio is None or not d.startswith("mic_probe_dev"):
            if d.startswith("mic_probe_dev"):
                print(f"{d}", flush=True)
            continue
        levels, live, distinct = summarize(audio, rate)
        n_mics = len(set(sum((list(map(int, p.split()[0].split("-"))) for p in distinct), [])))
        print(f"{d}\n    levels {[round(l) for l in levels]}  with sound: {live}  "
              f"different pairs: {distinct or 'none'}", flush=True)
        if n_mics > best[0]:
            best = (n_mics, d)
        mp.save_wav(re.sub(r"[^A-Za-z0-9_=-]+", "_", d)[:120], audio, rate)
    print(f"\nMOST SEPARATE MICROPHONES: {best[0]}  ({best[1]})", flush=True)
    print(f"device list and config files saved in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
