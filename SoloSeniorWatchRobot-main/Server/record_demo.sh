#!/usr/bin/env bash
# Demo video of the voice follow: Zenbo's camera with the voice banner and
# compass (saved by the server, so the window may be covered) and the sound
# the reSpeaker hears (saved by tests/respeaker_wifi_bridge.py), joined into
# one MP4. Needs run_voice_follow.sh running (RESPEAKER=wifi for the sound).
#   ./record_demo.sh            -> ~/Videos/zenbo_voice_demo_<date-time>.mp4
#   ./record_demo.sh name.mp4
# Ctrl+C stops the recording.
set -uo pipefail

out="${1:-$HOME/Videos/zenbo_voice_demo_$(date +%Y%m%d_%H%M%S).mp4}"
out="$(realpath -m "$out")"
mkdir -p "$(dirname "$out")"
base="${out%.mp4}"
request="$HOME/SoloSeniorWatchRobot_build/record_demo.path"

echo "$base" > "$request"
echo "Recording... (Ctrl+C to stop)"
trap ':' INT
sleep infinity &
wait $! 2>/dev/null
kill $! 2>/dev/null
trap - INT
rm -f "$request"
echo "Stopping..."
sleep 1.5   # the server and the bridge notice within 0.5 s

if [[ ! -s "$base.avi" ]]; then
    echo "No video was saved: is the server (run_voice_follow.sh) running?"
    exit 1
fi
if [[ -s "$base.wav" ]]; then
    ffmpeg -hide_banner -loglevel error -y -i "$base.avi" -i "$base.wav" -c:v libx264 -preset veryfast \
        -crf 22 -pix_fmt yuv420p -c:a aac -b:a 128k -shortest -movflags +faststart "$out"
else
    echo "(no reSpeaker sound saved: the video has no audio)"
    ffmpeg -hide_banner -loglevel error -y -i "$base.avi" -c:v libx264 -preset veryfast \
        -crf 22 -pix_fmt yuv420p -movflags +faststart "$out"
fi
rm -f "$base.avi" "$base.wav"
echo "Saved $out ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" | cut -d. -f1) s)"
