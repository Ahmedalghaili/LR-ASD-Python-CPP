"""Live test console for Zenbo (Zenbo Nurse Helper V2).

One window on the PC that shows, live:
  - the robot camera
  - whether the robot app is connected (camera FPS, audio, messages)
  - the microphone level the PC receives ("PC hears you")
  - whether Zenbo itself is listening and heard you, with the direction (DOA)
  - every message from the robot
  - a line that tells you what to do next
and lets you drive the robot with the keyboard.

Keys (click the window first):
  L  Zenbo listens once (speakAndListen)      S  stop listening
  T  turn to the last voice (lookAtUser)      A  auto: listen + turn, on/off
  W  "Hey Zenbo" wake word on/off
  Z / X  head left / right (15 deg)            R / F  head up / down
  B / N  body left / right (20 deg)            C  head to centre
  Q  quit

Stop the LR-ASD server first (same ports). Then:
  python3 zenbo_console.py
and connect Zenbo Nurse Helper V2 to this PC (port 8895).
"""
import queue
import threading
import time

import cv2
import numpy as np

import zenbo_poc as poc

W, H = 1280, 720
CAM_W, CAM_H = 640, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE, GREY, CYAN = (80, 220, 80), (60, 60, 230), (0, 220, 255), (240, 240, 240), (150, 150, 150), (230, 200, 60)


class ColorLink(poc.Link):
    """Link that also keeps the newest colour JPEG for display."""

    def __init__(self, *a, **k):
        self.last_jpeg = None
        super().__init__(*a, **k)

    def _images(self, conn):
        try:
            for payload in self._frames(conn):
                msg = poc.pb.RobotToServerMessage()
                msg.ParseFromString(payload)
                if not msg.jpegdata:
                    continue
                self.last_jpeg = msg.jpegdata
                img = cv2.imdecode(np.frombuffer(msg.jpegdata, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
                now = time.time()
                with self.lock:
                    self.frames.append((now, small))
                    self.frames = self.frames[-100:]
                    self.frame_times.append(now)
                    self.frame_times = self.frame_times[-500:]
        except OSError:
            pass

    def send(self, **fields):
        # The console must never block on a reconnect wait.
        cmd = poc.pb.RobotCommand(**fields)
        with self.lock:
            sock = self.cmd_sock
        if sock is None:
            return False
        try:
            sock.sendall(poc.HEAD + cmd.SerializeToString() + poc.TAIL)
            return True
        except OSError:
            with self.lock:
                self.cmd_sock = None
            return False


class Console:
    def __init__(self):
        self.link = ColorLink("0.0.0.0")
        self.commands = queue.Queue()
        self.log = []                 # (t, text, colour)
        self.seen_msgs = 0
        self.head_yaw, self.head_pitch = 0, 30
        self.auto = False
        self.wake_word = False
        self.listening_since = None   # our speakAndListen that has not finished
        self.last_voice_t = 0         # last voice event from Zenbo
        self.last_vad = ""
        self.last_doa = None
        self.last_doa_t = 0
        self.last_turn_t = 0
        self.audio_status = ""        # the app's own microphone report
        self.audio_status_t = 0
        threading.Thread(target=self._sender, daemon=True).start()
        self.add_log("waiting for Zenbo Nurse Helper V2 on port 8895 ...", GREY)

    # ------------------------------------------------------------------ robot
    def _sender(self):
        while True:
            fields, label = self.commands.get()
            ok = self.link.send(**fields)
            self.add_log(("> " if ok else "> NOT SENT (not connected): ") + label, CYAN if ok else RED)
            time.sleep(0.3)  # keep commands apart (older app versions)

    def do(self, label, **fields):
        self.commands.put((fields, label))

    def add_log(self, text, colour=WHITE):
        self.log.append((time.time(), text, colour))
        self.log = self.log[-14:]

    def poll_messages(self):
        with self.link.lock:
            new = self.link.messages[self.seen_msgs:]
            self.seen_msgs = len(self.link.messages)
        for t, m in new:
            d = m["description"]
            vad = m.get("vad_status", "")
            if d in ("onVoiceDetect", "onEventUserUtterance", "onListenResult"):
                self.last_voice_t = t
                self.last_vad = vad or d
                if vad == "onDOA" and "doa" in m:
                    self.last_doa, self.last_doa_t = m["doa"], t
                    self.add_log(f"< Zenbo heard you: direction (DOA) = {m['doa']}", GREEN)
                else:
                    self.add_log(f"< {d} {vad} " + (f"doa={m['doa']}" if "doa" in m else ""), WHITE)
            elif "DS_SPEAK_AND_LISTEN" in d:
                if "ACTIVE" in d:
                    self.listening_since = self.listening_since or t
                elif "SUCCEED" in d or "FAILED" in d:
                    self.listening_since = None
                self.add_log("< " + d.replace("robot_state ", "")[:70], GREY)
            elif d.startswith("audio_status"):
                if d.split(" reads_ok")[0] != self.audio_status.split(" reads_ok")[0]:
                    self.add_log("< " + d, YELLOW)
                self.audio_status, self.audio_status_t = d, t
            elif d.startswith("camera_"):
                self.add_log("< CAMERA PROBLEM: " + d, RED)
            elif d.startswith(("poc_result", "robot_state")):
                self.add_log("< " + d.replace("robot_state ", "")[:70], GREY)
            else:
                self.add_log("< " + d[:70], WHITE)

    def on_connect(self):
        self.do("report robot states", poc_action="states_on")
        self.do("head to centre", yaw=0, pitch=30, headspeed=2)
        self.head_yaw, self.head_pitch = 0, 30

    def listen(self):
        self.do("Zenbo: listen", poc_action="speak_and_listen")
        self.listening_since = time.time()

    def auto_step(self, now):
        if not self.auto:
            return
        if self.last_doa is not None and self.last_doa_t > self.last_turn_t:
            self.turn_to_voice()
            return
        if now - self.last_turn_t < 4:
            return  # let the turn finish
        if self.listening_since is None:
            self.listen()
        elif now - self.listening_since > 12:
            self.do("Zenbo: stop listening (timeout)", poc_action="stop_listen")
            self.listening_since = None

    def turn_to_voice(self):
        if self.last_doa is None:
            self.add_log("no voice direction yet: press L and speak", YELLOW)
            return
        self.do("stop listening", poc_action="stop_listen")
        self.do(f"turn to voice: lookAtUser({self.last_doa})", look_at_user_doa=float(self.last_doa))
        self.listening_since = None
        self.last_turn_t = time.time()

    # ------------------------------------------------------------------ keys
    def key(self, k):
        k = chr(k).lower() if 0 <= k < 256 else ""
        if k == "l":
            self.listen()
        elif k == "s":
            self.do("Zenbo: stop listening", poc_action="stop_listen")
            self.listening_since = None
        elif k == "t":
            self.turn_to_voice()
        elif k == "a":
            self.auto = not self.auto
            self.add_log(f"AUTO mode {'ON: speak, Zenbo turns to you' if self.auto else 'OFF'}", YELLOW)
            if self.auto:
                self.last_turn_t = 0
        elif k == "w":
            self.wake_word = not self.wake_word
            self.do(f"wake word {'on' if self.wake_word else 'off'}",
                    poc_action="voice_trigger_on" if self.wake_word else "voice_trigger_off")
        elif k in "zx":
            self.head_yaw = max(-45, min(45, self.head_yaw + (15 if k == "z" else -15)))  # + = left
            self.do(f"head yaw {self.head_yaw:+d}", yaw=self.head_yaw, pitch=self.head_pitch, headspeed=2)
        elif k in "rf":
            self.head_pitch = max(-15, min(55, self.head_pitch + (10 if k == "r" else -10)))
            self.do(f"head pitch {self.head_pitch}", yaw=self.head_yaw, pitch=self.head_pitch, headspeed=2)
        elif k in "bn":
            deg = 20 if k == "b" else -20  # + = left
            self.do(f"body turn {deg:+d}", degree=deg)
        elif k == "c":
            self.head_yaw, self.head_pitch = 0, 30
            self.do("head to centre", yaw=0, pitch=30, headspeed=2)
        return k == "q"

    # ------------------------------------------------------------------ draw
    def status(self, now):
        with self.link.lock:
            frames = [t for t in self.link.frame_times if now - t < 2]
            audio = [(t, d) for t, d in self.link.audio if now - t < 1]
            connected = self.link.cmd_sock is not None
        return connected, len(frames) / 2.0, audio

    def instruction(self, now, connected, fps, audio):
        if not connected:
            return "Open Zenbo Nurse Helper V2 and connect to this PC (port 8895).", YELLOW
        if fps < 1:
            return "Connected, but no camera images: check camera permission of V2.", RED
        if not audio:
            st = self.audio_status if now - self.audio_status_t < 12 else ""
            if "permission=NO" in st:
                return "Robot: microphone permission is OFF. Settings > Apps > V2 > Permissions.", RED
            if "recorder=FAILED" in st or "recording=NO" in st:
                return "Robot: microphone busy/not recording (Zenbo may hold it). Restart Zenbo.", RED
            if "read_errors" in st and "reads_ok=0" in st:
                return "Robot: microphone read errors (" + st.split("last_error=")[-1] + "). Restart Zenbo.", RED
            if not st:
                return "No microphone sound, and no mic report: install the newest V2 app.", RED
            return "No microphone sound from the app.", RED
        if self.listening_since and now - self.last_voice_t > 3:
            return "Zenbo is LISTENING: speak now (a full sentence).", GREEN
        if now - self.last_doa_t < 4 and self.last_doa is not None:
            return f"Zenbo heard you from direction {self.last_doa}. Press T to turn (or A for auto).", GREEN
        if self.auto:
            return "AUTO: speak from any side; Zenbo listens and turns to you.", GREEN
        return "Press L, then speak. Or press A for automatic listen + turn.", WHITE

    def draw(self):
        now = time.time()
        connected, fps, audio = self.status(now)
        canvas = np.full((H, W, 3), 25, np.uint8)
        # Camera
        cam = None
        if self.link.last_jpeg is not None:
            cam = cv2.imdecode(np.frombuffer(self.link.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if cam is not None:
            canvas[20:20 + CAM_H, 20:20 + CAM_W] = cv2.resize(cam, (CAM_W, CAM_H))
        else:
            cv2.putText(canvas, "no camera image", (230, 260), FONT, 0.8, GREY, 2)
        cv2.rectangle(canvas, (20, 20), (20 + CAM_W, 20 + CAM_H), GREY, 1)

        # Instruction
        text, colour = self.instruction(now, connected, fps, audio)
        cv2.rectangle(canvas, (20, 515), (W - 20, 560), (50, 50, 50), -1)
        cv2.putText(canvas, text, (30, 546), FONT, 0.8, colour, 2)

        # Status panel
        x = 690
        y = 50
        def row(label, value, ok):
            nonlocal y
            cv2.putText(canvas, label, (x, y), FONT, 0.6, GREY, 1)
            cv2.putText(canvas, value, (x + 210, y), FONT, 0.6, GREEN if ok else RED, 2)
            y += 32
        row("Robot app", "CONNECTED" if connected else "not connected", connected)
        row("Camera", f"{fps:.0f} FPS", fps >= 5)
        level = max((d for _, d in audio), default=-120)
        row("Microphone to PC", f"{len(audio) * 1:.0f} chunks/s" if audio else "NO AUDIO", bool(audio))
        # Level bar: -60 dB .. 0 dB
        frac = min(max((level + 60) / 60, 0), 1)
        cv2.rectangle(canvas, (x + 210, y - 18), (x + 510, y - 2), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x + 210, y - 18), (x + 210 + int(300 * frac), y - 2),
                      GREEN if level > -35 else GREY, -1)
        cv2.putText(canvas, "PC hears you" if level > -35 else "quiet", (x, y - 4), FONT, 0.6,
                    GREEN if level > -35 else GREY, 1)
        y += 22
        listening = self.listening_since is not None
        row("Zenbo listening", "YES - speak" if listening else "no (press L)", listening)
        heard = now - self.last_voice_t < 5
        row("Zenbo heard voice", self.last_vad if heard else "-", heard)
        row("Wake word / Auto", f"{'on' if self.wake_word else 'off'} / {'ON' if self.auto else 'off'}", self.auto)
        row("Head yaw / pitch", f"{self.head_yaw:+d} / {self.head_pitch}", True)
        st = self.audio_status.replace("audio_status ", "") if now - self.audio_status_t < 12 else "no report"
        cv2.putText(canvas, "Mic on robot: " + st[:60], (x, y), FONT, 0.45,
                    GREEN if "recording=yes" in st else RED, 1)
        y += 20

        # Direction compass (raw Zenbo DOA; robot faces up)
        cx, cy, r = x + 470, 395, 70
        cv2.circle(canvas, (cx, cy), r, GREY, 1)
        cv2.putText(canvas, "front", (cx - 22, cy - r - 8), FONT, 0.45, GREY, 1)
        cv2.circle(canvas, (cx, cy), 8, WHITE, -1)
        if self.last_doa is not None:
            a = np.deg2rad(self.last_doa)
            end = (int(cx - r * np.sin(a)), int(cy - r * np.cos(a)))  # assumes + = left
            cv2.arrowedLine(canvas, (cx, cy), end, GREEN if now - self.last_doa_t < 5 else GREY, 3)
            cv2.putText(canvas, f"DOA {self.last_doa}", (cx - 40, cy + r + 20), FONT, 0.55, WHITE, 1)
        cv2.putText(canvas, "last voice direction", (x, 350), FONT, 0.55, GREY, 1)
        cv2.putText(canvas, "(raw Zenbo value;", (x, 375), FONT, 0.45, GREY, 1)
        cv2.putText(canvas, " arrow assumes + = left)", (x, 395), FONT, 0.45, GREY, 1)

        # Log
        y = 590
        for t, line, col in self.log[-6:]:
            cv2.putText(canvas, time.strftime("%H:%M:%S ", time.localtime(t)) + line[:110], (25, y),
                        FONT, 0.5, col, 1)
            y += 20
        cv2.putText(canvas, "Keys: L listen  S stop  T turn to voice  A auto  W wake word  "
                    "Z/X head L/R  R/F head up/down  B/N body L/R  C centre  Q quit",
                    (25, H - 8), FONT, 0.5, YELLOW, 1)
        return canvas

    def run(self):
        cv2.namedWindow("Zenbo console", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Zenbo console", W, H)
        was_connected = False
        while True:
            now = time.time()
            connected = self.link.cmd_sock is not None
            if connected and not was_connected:
                self.add_log("Zenbo connected", GREEN)
                self.on_connect()
            if was_connected and not connected:
                self.add_log("Zenbo DISCONNECTED - reconnect the app", RED)
            was_connected = connected
            self.poll_messages()
            if self.listening_since and now - self.listening_since > 15:
                self.listening_since = None  # no finish report; treat as ended
            self.auto_step(now)
            cv2.imshow("Zenbo console", self.draw())
            k = cv2.waitKey(50)
            if k != -1 and self.key(k & 0xFF):
                break
        self.link.send(poc_action="stop_listen")
        self.link.send(yaw=0, pitch=30, headspeed=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Console().run()
