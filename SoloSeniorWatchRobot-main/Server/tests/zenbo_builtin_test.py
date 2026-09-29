"""Try Zenbo Junior's built-in (ASUS) functions from this PC with PyZenbo.

Requirements
  * PyZenbo (Apache-2.0) from the ASUS developer site, or the 1.0.7 copy in
    github.com/marianhlavac/zenbo-junior-hide-and-seek (folder pyzenbo/).
    Pass its folder with --pyzenbo or install it (python setup.py install).
  * ZenboLab running on the robot, robot and PC on the same Wi-Fi
    (PyZenbo uses TCP port 55555).

Steps (run them in this order; only `look` and `track` move the robot):
  hello  connect, show a face and say hello              (no movement)
  doa    make Zenbo listen and print the direction (DOA)
         of each utterance; speak from different sides   (no movement)
  look   like doa, then turn toward each speaker with
         Zenbo's built-in utility.look_at_user(doa)      (ROBOT TURNS)
  track  Zenbo's built-in utility.track_face for N s     (ROBOT MOVES)

Examples:
  python zenbo_builtin_test.py 192.168.1.65 hello --pyzenbo ~/pyzenbo
  python zenbo_builtin_test.py 192.168.1.65 doa --seconds 60 --pyzenbo ~/pyzenbo
Press Ctrl+C at any time; all robot commands are cancelled on exit.
"""
import argparse
import json
import os
import sys
import time

# Any 32-hex domain id works for listening; this one is from the PyZenbo demo.
DOMAIN = "E7AABB554ACB414C9AB9BF45E7FA8AD9"


def find_key(obj, key, depth=4):
    """Return obj[key] at any nesting level (values may be JSON strings)."""
    if depth < 0:
        return None
    if isinstance(obj, str) and obj.startswith("{"):
        try:
            obj = json.loads(obj)
        except ValueError:
            return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key, depth - 1)
            if found is not None:
                return found
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_ip")
    ap.add_argument("step", choices=["hello", "doa", "look", "track"])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--pyzenbo", help="folder that contains the pyzenbo package")
    ap.add_argument("--log", default="zenbo_builtin_test.jsonl",
                    help="every listen event is appended here as JSON")
    args = ap.parse_args()

    if args.pyzenbo:
        sys.path.insert(0, os.path.expanduser(args.pyzenbo))
    import pyzenbo
    from pyzenbo.modules.dialog_system import RobotFace

    print(f"PyZenbo {pyzenbo.__version__}: connecting to {args.robot_ip}:55555 ...")
    sdk = pyzenbo.connect(args.robot_ip)
    # PyZenbo does not raise when the robot refuses the connection; check it.
    from pyzenbo.modules.socket_state_machine import SOCKET_STATE
    if SOCKET_STATE["Connected"] not in sdk.get_connection_state():
        sdk.release()
        sys.exit(f"Could not connect to {args.robot_ip}:55555. Start ZenboLab on the robot "
                 "and make sure the robot and this PC are on the same Wi-Fi.")
    print("connected")
    log = open(args.log, "a")
    last = {"doa": None, "utterance": None}

    def on_listen(result):
        entry = {"t": round(time.time(), 2), "raw": result}
        log.write(json.dumps(entry, default=str) + "\n")
        log.flush()
        doa = find_key(result, "doa")
        vad = find_key(result, "vad_status")
        level = find_key(result, "sound_level")
        utterance = find_key(result, "user_utterance")
        if doa is not None:
            last["doa"] = doa
        if utterance:
            last["utterance"] = utterance
        print(f"  listen event: vad_status={vad} sound_level={level} doa={doa}"
              + (f" utterance={utterance!r}" if utterance else ""))

    try:
        if args.step == "hello":
            sdk.robot.set_expression(RobotFace.INTERESTED, "Hello, the built-in test works.",
                                     timeout=15)
            print("OK: if Zenbo showed a face and spoke, PyZenbo works.")
            return

        if args.step in ("doa", "look"):
            sdk.robot.register_listen_callback(DOMAIN, on_listen)
            end = time.time() + args.seconds
            print("Speak a short sentence, then move to another side and speak again.")
            while time.time() < end:
                last["doa"] = last["utterance"] = None
                # An empty sentence makes Zenbo listen directly for one utterance.
                sdk.robot.speak_and_listen("", timeout=12)
                print(f"utterance done: doa={last['doa']} text={last['utterance']!r}")
                if args.step == "look" and last["doa"] is not None:
                    print(f"  -> look_at_user({last['doa']})  (robot turns)")
                    sdk.utility.look_at_user(last["doa"], timeout=15)
            return

        if args.step == "track":
            print(f"track_face for {args.seconds:.0f} s (robot follows a face with head/body)")
            sdk.utility.track_face(enable_preview=False, large_preview=False)
            time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        for stop in (lambda: sdk.robot.stop_speak_and_listen(timeout=5),
                     lambda: sdk.cancel_command_all(timeout=5)):
            try:
                stop()
            except Exception as exc:  # the robot may already be idle
                print("cleanup:", exc)
        sdk.release()
        log.close()
        print(f"listen events saved in {args.log}")


if __name__ == "__main__":
    main()
