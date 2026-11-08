"""Bridge for the reSpeaker on Wi-Fi (tests/respeaker_wifi_audio/*.ino on its
XIAO ESP32S3): the board's voice goes to the voice gate (VAD + CAM++) and its
direction to the server.

  board -> UDP 5011 "AUD" audio (16 kHz int16)  -> 127.0.0.1:5008 voice_gate.py
  board -> UDP 5007 "DOA <doa> <speech> <free>" -> 127.0.0.1:5006 server, as
           "RAW <free-running beam deg> <speech>" (the beam follows voices;
           DOA_VALUE got stuck inside Zenbo, tests/respeaker_compare_doa.py)
The board streams to whoever broadcasts "HELLO" to UDP 5010, so this sends
one every 2 s. Run the server with ACTIVE_SPEAKER_VOICE_GATE=respeaker.
The PC firewall must let UDP 5007 and 5011 in: sudo ufw allow 5007,5011/udp
While ~/SoloSeniorWatchRobot_build/record_demo.path exists (record_demo.sh),
the board's audio is also saved to <path in the file>.wav.
"""
import os
import select
import socket
import time
import wave

HELLO_PORT, AUDIO_PORT, DOA_PORT = 5010, 5011, 5007
GATE = ("127.0.0.1", 5008)
SERVER = ("127.0.0.1", 5006)
DEMO_REQUEST = os.path.expanduser("~/SoloSeniorWatchRobot_build/record_demo.path")


def demo_path():
    try:
        with open(DEMO_REQUEST) as f:
            return f.read().strip() or None
    except OSError:
        return None


def main():
    hello = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hello.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio.bind(("", AUDIO_PORT))
    doa = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    doa.bind(("", DOA_PORT))
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("looking for the reSpeaker on Wi-Fi ...", flush=True)
    board, last_hello, last_stats = None, 0.0, time.time()
    packets = lost = readings = speech = 0
    expected = None
    demo, demo_wav, last_demo_check = None, None, 0.0
    while True:
        now = time.time()
        if now - last_demo_check >= 0.5:
            last_demo_check = now
            path = demo_path()
            if path != demo:
                if demo_wav:
                    demo_wav.close()
                    print(time.strftime("%H:%M:%S"), f"saved {demo}.wav", flush=True)
                demo, demo_wav = path, None
                if path:
                    demo_wav = wave.open(path + ".wav", "wb")
                    demo_wav.setnchannels(1)
                    demo_wav.setsampwidth(2)
                    demo_wav.setframerate(16000)
        if now - last_hello >= 2:
            hello.sendto(b"HELLO", ("255.255.255.255", HELLO_PORT))
            last_hello = now
        ready, _, _ = select.select([audio, doa], [], [], 0.5)
        for sock in ready:
            data, addr = sock.recvfrom(2048)
            if addr[0] != board:
                print(time.strftime("%H:%M:%S"), "reSpeaker found at", addr[0], flush=True)
                board = addr[0]
            if sock is audio and data[:3] == b"AUD":
                seq = data[3] | data[4] << 8
                if expected is not None:
                    lost += (seq - expected) % 65536
                expected = (seq + 1) % 65536
                packets += 1
                out.sendto(data[5:], GATE)
                if demo_wav:
                    demo_wav.writeframes(data[5:])
            elif sock is doa:
                parts = data.decode(errors="replace").split()
                if len(parts) == 4 and parts[0] == "DOA":
                    readings += 1
                    if parts[2] == "1":
                        speech += 1
                    if int(parts[3]) >= 0:
                        out.sendto(f"RAW {parts[3]} {parts[2]}".encode(), SERVER)
        if now - last_stats >= 5:
            if board:
                print(time.strftime("%H:%M:%S"), f"audio {packets / (now - last_stats):.0f} packets/s "
                      f"(50 expected), lost {lost}; direction readings {readings}, with speech {speech}",
                      flush=True)
            packets = lost = readings = speech = 0
            last_stats = now


if __name__ == "__main__":
    main()
