#!/usr/bin/env bash
# Zenbo turns toward whoever talks:
#   1. Silero VAD (tests/voice_gate.py): real speech, or just noise?
#   2. CAM++ voiceprint (same script): an enrolled person, or unknown?
#   3. reSpeaker XVF3800 on Zenbo's USB (free-running beam): where? Turn there.
# Starts the voice gate and the server; Ctrl+C stops both.
#
# Enroll a person (with this script already running in another terminal):
#   ./run_voice_follow.sh enroll Ahmed
# then the person talks until 10 s of speech are heard.
# KNOWN_ONLY=1 ./run_voice_follow.sh   turns only for enrolled people.
# RESPEAKER=wifi ./run_voice_follow.sh  the board sends its own voice and
#   direction over Wi-Fi (tests/respeaker_wifi_audio, I2S firmware) through
#   tests/respeaker_wifi_bridge.py, instead of Zenbo's USB and microphone.
# Settings for the board as mounted now: front = array angle 180, mirror on,
# Zenbo's own noise at array angle 90 (tests/respeaker_compare_doa.py).
set -euo pipefail

cd "$(dirname "$0")"
python="$(cd ../.. && pwd)/.venv/bin/python"
logs="$HOME/SoloSeniorWatchRobot_build"
gate_pid_file="$logs/voice_gate.pid"
bridge_pid=""

start_gate() {
    "$python" -u tests/voice_gate.py >> "$logs/voice_gate.log" 2>&1 &
    echo $! > "$gate_pid_file"
}

stop_gate() {
    [[ -f "$gate_pid_file" ]] && kill "$(cat "$gate_pid_file")" 2>/dev/null || true
    rm -f "$gate_pid_file"
}

if [[ "${1:-}" == "enroll" ]]; then
    [[ -n "${2:-}" ]] || { echo "usage: $0 enroll NAME"; exit 1; }
    echo "$2" > "$logs/speaker/enroll_request"
    echo "Talk normally near the microphone. Progress (Ctrl+C when it says Saved):"
    tail -n 0 -f "$logs/voice_gate.log"
    exit 0
fi

stop_gate
start_gate
gate_audio=1
if [[ "${RESPEAKER:-usb}" == "wifi" ]]; then
    "$python" -u tests/respeaker_wifi_bridge.py >> "$logs/respeaker_wifi.log" 2>&1 &
    bridge_pid=$!
    gate_audio=respeaker
    echo "reSpeaker Wi-Fi bridge log: $logs/respeaker_wifi.log"
fi
trap 'stop_gate; [[ -n "$bridge_pid" ]] && kill "$bridge_pid" 2>/dev/null || true' EXIT
echo "voice gate log: $logs/voice_gate.log   server log: $logs/server_voice.log"

ACTIVE_SPEAKER_FOLLOW=robot \
ACTIVE_SPEAKER_DOA=respeaker \
ACTIVE_SPEAKER_RESPEAKER_MIRROR=1 \
ACTIVE_SPEAKER_RESPEAKER_NOISE=90 \
ACTIVE_SPEAKER_VOICE_GATE="$gate_audio" \
ACTIVE_SPEAKER_KNOWN_ONLY="${KNOWN_ONLY:-0}" \
SSWR_ROBOT_SILENT=1 \
ACTIVE_SPEAKER_LOGIT=-1.0 \
ACTIVE_SPEAKER_ACQUIRE_MS=200 \
LR_ASD_DETECTOR=scrfd \
LR_ASD_CROP_ALL_FRAMES=1 \
LR_ASD_VERBOSE_LOG=1 \
./run_with_lrasd.sh build/SoloSeniorWatchRobot --SettingFile json/ROGG16_SSWR.json \
    2>&1 | tee "$logs/server_voice.log"
