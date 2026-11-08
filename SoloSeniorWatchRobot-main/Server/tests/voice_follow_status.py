"""Live status of run_voice_follow.sh in plain words: is Zenbo connected, does
the reSpeaker send, whose voice is heard, where Zenbo turns.

  python tests/voice_follow_status.py
Reads the logs in ~/SoloSeniorWatchRobot_build (server_voice.log,
voice_gate.log, respeaker_wifi.log) and the open connections on port 8898.
"""
import json
import os
import re
import subprocess
import time

LOGS = os.path.expanduser("~/SoloSeniorWatchRobot_build")


class Follow:
    """New lines of a log file."""
    def __init__(self, name):
        self.path = os.path.join(LOGS, name)
        self.f = None

    def lines(self):
        if self.f is None:
            try:
                self.f = open(self.path, errors="replace")
                self.f.seek(0, os.SEEK_END)
            except OSError:
                return []
        out = []
        while True:
            line = self.f.readline()
            if not line:
                return out
            out.append(line.rstrip())


def say(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def zenbo_address():
    """IP of the robot connected to port 8898, or None."""
    out = subprocess.run(["ss", "-tn", "state", "established", "( sport = :8898 )"],
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+\.\d+\.\d+\.\d+):\d+\s*$", out, re.M)
    return m.group(1) if m else None


def side(doa):
    if abs(doa) <= 20:
        return "in front"
    if abs(doa) >= 160:
        return "behind"
    return f"{abs(doa)} deg to the {'left' if doa > 0 else 'right'}"


def main():
    server, gate, wifi = Follow("server_voice.log"), Follow("voice_gate.log"), Follow("respeaker_wifi.log")
    print("Voice follow status (Ctrl+C to quit)\n", flush=True)
    robot, board_ok, last_check = "start", None, 0.0
    while True:
        if time.time() - last_check >= 2:
            last_check = time.time()
            ip = zenbo_address()
            if ip != robot:
                say(f"Zenbo: CONNECTED ({ip})" if ip else "Zenbo: not connected - open the app on Zenbo")
                robot = ip
        for line in wifi.lines():
            m = re.search(r"audio (\d+) packets/s.*direction readings (\d+)", line)
            if m:
                ok = int(m.group(1)) > 40 and int(m.group(2)) > 10
                if ok != board_ok:
                    say("reSpeaker: OK (sound and direction arriving)" if ok
                        else "reSpeaker: NO DATA - is the board powered and on Wi-Fi?")
                    board_ok = ok
            elif "found at" in line:
                say("reSpeaker: " + line.split(" ", 1)[1])
        for line in gate.lines():
            m = re.search(r"voice: (\S+) \(score ([\d.]+)\)", line)
            if m:
                say(f"Heard: {m.group(1).upper() if m.group(1) == 'unknown' else m.group(1)} "
                    f"(score {m.group(2)})")
            elif "Saved" in line or "Enrolling" in line:
                say(line.split(" ", 1)[1])
        for line in server.lines():
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("component") == "ActiveSpeaker" and e.get("event") == "turn_to_voice":
                say(f"Zenbo TURNS to the voice, {side(int(e.get('voice_doa', 0)))}")
            elif e.get("component") == "ZenboApp" and "MOTION_MOVE_BODY" in e.get("message", "") \
                    and "SUCCEED" in e.get("message", ""):
                say("  turn done")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
