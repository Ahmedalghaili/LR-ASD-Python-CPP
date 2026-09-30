"""Can the PC compute the voice direction from Zenbo's own microphones?

Needs Zenbo Nurse Helper V2 with the microphone probe (v2.2026.09.30-micprobe).
Stop the LR-ASD server and the console first (same ports), then:
  python3 zenbo_mic_probe.py
and connect V2 to this PC. Zenbo tells you what to do.

1. Probe: the app records 1 s with every audio source x channel setup x
   rate it can open. For each, this tool checks whether the channels are
   really different microphones (not copies of one) and how strong they are.
2. Direction: with the best multi-channel setup, Zenbo asks you to talk from
   its front, left, right and back. For each position the tool measures the
   time difference of arrival between channel pairs (GCC-PHAT). If left and
   right give clearly different (opposite) delays, the PC can compute the
   direction of a voice itself.

Every recording is saved as a WAV file in mic_probe_out/.
Report: mic_probe_report.json.
"""
import json
import os
import threading
import time
import wave

import numpy as np

import zenbo_poc as poc

OUT = "mic_probe_out"


class ProbeLink(poc.Link):
    """Link that keeps the probe recordings."""

    def __init__(self, *a, **k):
        self.recordings = []   # (description, np.int16 array [samples, channels], rate)
        self.rec_lock = threading.Lock()
        super().__init__(*a, **k)

    def _messages(self, conn):
        try:
            for payload in self._frames(conn):
                msg = poc.pb.RobotToServerMessage()
                msg.ParseFromString(payload)
                d = msg.description
                if d.startswith(("mic_probe", "mic_record")):
                    audio = None
                    if msg.audio_probe and msg.audio_probe_channels > 0:
                        a = np.frombuffer(msg.audio_probe, np.int16)
                        ch = msg.audio_probe_channels
                        audio = a[: len(a) // ch * ch].reshape(-1, ch)
                    with self.rec_lock:
                        self.recordings.append((d, audio, msg.audio_probe_rate))
                    print(f"[robot] {d}", flush=True)
                elif not d.startswith(("audio_status", "robot_state")):
                    print(f"[robot] {d}", flush=True)
                with self.lock:
                    self.messages.append((time.time(), {"description": d}))
        except OSError:
            pass


def dbfs(x):
    x = x.astype(np.float64)
    return 20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1.0) / 32768)


def gcc_phat(a, b, rate, max_ms=1.5, frame=2048):
    """Median time difference of arrival (microseconds, + = sound reaches b
    first) over loud frames, and how consistent the frames are."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    max_shift = int(rate * max_ms / 1000)
    lags = []
    energy = []
    for start in range(0, len(a) - frame, frame // 2):
        x = a[start:start + frame]
        y = b[start:start + frame]
        energy.append(np.sqrt(np.mean(x ** 2)))
        X = np.fft.rfft(x * np.hanning(frame), 4 * frame)
        Y = np.fft.rfft(y * np.hanning(frame), 4 * frame)
        R = X * np.conj(Y)
        R /= np.abs(R) + 1e-9
        cc = np.fft.irfft(R, 4 * 4 * frame)          # 4x interpolation
        n = len(cc)
        shift = max_shift * 4
        window = np.concatenate((cc[-shift:], cc[:shift + 1]))
        lags.append((np.argmax(window) - shift) / 4.0)
    if not lags:
        return None, 0.0
    lags = np.array(lags)
    energy = np.array(energy)
    loud = energy >= np.percentile(energy, 60)       # frames with the voice
    sel = lags[loud] if loud.any() else lags
    median = float(np.median(sel))
    consistency = float(np.mean(np.abs(sel - median) <= 1.0))
    return median / rate * 1e6, consistency


def analyse(audio, rate):
    ch = audio.shape[1]
    info = {"channels": ch, "rate": rate,
            "level_dbfs": [round(dbfs(audio[:, i]), 1) for i in range(ch)]}
    pairs = []
    for i in range(ch):
        for j in range(i + 1, ch):
            x, y = audio[:, i].astype(np.float64), audio[:, j].astype(np.float64)
            identical = bool(np.array_equal(audio[:, i], audio[:, j]))
            silent = max(np.std(x), np.std(y)) < 1
            corr = 0.0 if silent or np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
            pairs.append({"pair": [i, j], "identical": identical, "silent": bool(silent),
                          "correlation": round(corr, 4)})
    distinct = [p for p in pairs if not p["identical"] and not p["silent"] and p["correlation"] < 0.999]
    info["pairs"] = pairs
    info["distinct_pairs"] = len(distinct)
    return info


def save_wav(name, audio, rate):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype("<i2").tobytes())
    return path


def wait_for(link, prefix, count, timeout):
    end = time.time() + timeout
    while time.time() < end:
        with link.rec_lock:
            n = sum(1 for d, _, _ in link.recordings if d.startswith(prefix))
        if n >= count:
            return True
        time.sleep(0.3)
    return False


def main():
    link = ProbeLink("0.0.0.0")
    print("waiting for Zenbo Nurse Helper V2 on port 8895 ...", flush=True)
    link.connected.wait()
    time.sleep(3)
    report = {"probe": [], "direction": []}

    # 1. Probe every setup while someone talks.
    link.send(speak_sentence="Microphone test. Please keep talking near me for one minute.")
    time.sleep(6)
    link.send(poc_action="mic_probe")
    ok = wait_for(link, "mic_probe_done", 1, 240)
    print("probe finished" if ok else "probe did not finish in 4 minutes", flush=True)
    link.send(speak_sentence="Thank you.")

    best = None
    with link.rec_lock:
        recs = list(link.recordings)
    print("\nsource / setup / rate            ch  result      levels dBFS          distinct pairs")
    for d, audio, rate in recs:
        if not d.startswith("mic_probe ") or "source=" not in d:
            continue
        fields = dict(kv.split("=", 1) for kv in d.split(" ") if "=" in kv)
        row = {"description": d, **fields}
        if audio is not None and len(audio) > 0:
            info = analyse(audio, rate)
            row.update(info)
            row["wav"] = save_wav(f"probe_{fields['source']}_{fields['setup']}_{rate}", audio, rate)
            score = (info["distinct_pairs"], info["channels"], rate)
            if info["distinct_pairs"] > 0 and (best is None or score > best[0]):
                best = (score, fields["source"], fields["setup"], rate)
        report["probe"].append(row)
        print(f"{fields.get('source',''):20s} {fields.get('setup',''):7s} {fields.get('rate',''):6s} "
              f"{row.get('channels', fields.get('channels','-'))!s:3s} {fields.get('result','')[:11]:11s} "
              f"{str(row.get('level_dbfs', '-')):28s} {row.get('distinct_pairs', '-')}", flush=True)

    if best is None:
        print("\nRESULT: no setup gave two or more different microphone channels.\n"
              "The PC cannot compute the direction from this robot's microphones.", flush=True)
        report["result"] = "no multi-channel microphone access"
    else:
        _, source, setup, rate = best
        print(f"\nBEST: source={source} setup={setup} rate={rate} -> direction test", flush=True)
        report["best"] = {"source": source, "setup": setup, "rate": rate}
        positions = (("front", "Please stand in front of me, and keep talking."),
                     ("left", "Now stand on my left side, and keep talking."),
                     ("right", "Now stand on my right side, and keep talking."),
                     ("behind", "Now stand behind me, and keep talking."))
        for name, text in positions:
            print(f"\n=== {name}: {text}", flush=True)
            link.send(speak_sentence=text)
            time.sleep(7)
            before = len(link.recordings)
            link.send(poc_action=f"mic_record:{source}:{setup}:{rate}:4")
            if not wait_for_new(link, before, 20):
                print("  no recording received", flush=True)
                continue
            d, audio, r = link.recordings[-1]
            if audio is None:
                print("  " + d, flush=True)
                continue
            save_wav(f"direction_{name}", audio, r)
            tdoa = {}
            for i in range(audio.shape[1]):
                for j in range(i + 1, audio.shape[1]):
                    us, cons = gcc_phat(audio[:, i], audio[:, j], r)
                    tdoa[f"{i}-{j}"] = {"tdoa_us": None if us is None else round(us, 1),
                                        "consistency": round(cons, 2)}
            level = [round(dbfs(audio[:, i]), 1) for i in range(audio.shape[1])]
            report["direction"].append({"position": name, "levels_dbfs": level, "tdoa": tdoa})
            print(f"  levels {level}", flush=True)
            for k, v in tdoa.items():
                print(f"  pair {k}: delay {v['tdoa_us']} us  (consistency {v['consistency']})", flush=True)
        link.send(speak_sentence="The test is finished. Thank you.")
        report["result"] = "see direction delays per position"

    with open("mic_probe_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nreport saved in mic_probe_report.json, recordings in mic_probe_out/", flush=True)
    time.sleep(2)


def wait_for_new(link, before, timeout):
    end = time.time() + timeout
    while time.time() < end:
        with link.rec_lock:
            if len(link.recordings) > before and link.recordings[-1][0].startswith("mic_record"):
                return True
        time.sleep(0.3)
    return False


if __name__ == "__main__":
    main()
