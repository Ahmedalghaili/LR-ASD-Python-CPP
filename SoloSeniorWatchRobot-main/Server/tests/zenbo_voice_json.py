"""Record the raw JSON of Zenbo's voice events (onVoiceDetect,
onEventUserUtterance, listen results) as the robot sends them.

Needs Zenbo Nurse Helper V2 that forwards "zenbo_json" messages. Stop the
LR-ASD server first (same ports). Zenbo is kept listening (speakAndListen,
never interrupted) and asks the user to speak from the front, its left, its
right and behind. Every raw JSON is saved to zenbo_voice_json.txt.
"""
import sys
import time

import zenbo_poc as poc

OUT = "zenbo_voice_json.txt"


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 240
    link = poc.Link("0.0.0.0")
    print("waiting for Zenbo Nurse Helper V2 on port 8895 ...", flush=True)
    link.connected.wait()
    time.sleep(4)
    out = open(OUT, "w")

    def send(**fields):
        # Never block on a reconnect wait: skip while the app is away.
        with link.lock:
            connected = link.cmd_sock is not None
        if connected:
            try:
                sock = link.cmd_sock
                sock.sendall(poc.HEAD + poc.pb.RobotCommand(**fields).SerializeToString() + poc.TAIL)
                time.sleep(0.3)
                return True
            except OSError:
                with link.lock:
                    link.cmd_sock = None
        return False
    seen = 0
    t_end = time.time() + seconds
    positions = ["in front of me", "on my left side", "on my right side", "behind me"]
    connections = -1
    listen_at = 0
    listening = False
    position = 0
    next_position = 0
    while time.time() < t_end:
        with link.lock:
            new = link.messages[seen:]
            seen = len(link.messages)
            reconnected = link.connections != connections
            connections = link.connections
        if reconnected:
            send(poc_action="states_on")
            send(poc_action="stereo_off")
            listening = False
        for t, m in new:
            d = m["description"]
            if d.startswith("zenbo_json"):
                out.write(f"=== {time.strftime('%H:%M:%S', time.localtime(t))} {d}\n\n")
                out.flush()
                print(d.replace("\n", " ")[:160], flush=True)
            if d == "onEventUserUtterance" or ("DS_SPEAK_AND_LISTEN" in d and ("SUCCEED" in d or "FAILED" in d)):
                listening = False
        now = time.time()
        if now >= next_position and position < len(positions):
            if not send(speak_sentence=f"Please stand {positions[position]}, and talk to me."):
                time.sleep(0.5)
                continue  # app not connected (e.g. updating): ask again when it is back
            print(f"--- position: {positions[position]}", flush=True)
            position += 1
            next_position = now + 60
            listening = False
            listen_at = now + 5  # after the sentence
        if not listening and now >= listen_at:
            if send(poc_action="speak_and_listen"):
                listening = True
            listen_at = now
        elif listening and now - listen_at > 20:  # no end reported: start again, never stop it
            listening = False
            listen_at = now
        time.sleep(0.2)
    send(speak_sentence="Thank you. The test is finished.")
    out.close()
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
