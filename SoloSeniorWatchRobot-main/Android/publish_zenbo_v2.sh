#!/bin/bash
# Build Zenbo Nurse Helper V2 and publish it for the robot's self-update.
# The app checks http://<this PC>:8000/latest.json whenever it connects and
# opens Android's install screen for a newer version (tap "Install").
# Run serve_zenbo_updates.sh once so port 8000 serves the folder below.
set -e
cd "$(dirname "$0")"
UPDATES="${ZENBO_UPDATES_DIR:-$HOME/SoloSeniorWatchRobot_build/app_updates}"
CODE=$(( ( $(date +%s) - $(date -d 2026-01-01 +%s) ) / 60 ))   # always increasing
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
bash ./gradlew :robot:assembleZenboV2Debug -Pv2VersionCode=$CODE --console=plain -q
mkdir -p "$UPDATES"
cp robot/build/outputs/apk/ZenboV2/debug/robot-ZenboV2-debug.apk "$UPDATES/ZenboV2_latest.apk.tmp"
mv "$UPDATES/ZenboV2_latest.apk.tmp" "$UPDATES/ZenboV2_latest.apk"
NAME=$(grep -o "versionName '[^']*'" robot/build.gradle | sed -n 3p | cut -d"'" -f2)-zenbo-v2
printf '{"versionCode": %d, "versionName": "%s", "file": "ZenboV2_latest.apk"}\n' "$CODE" "$NAME" > "$UPDATES/latest.json"
echo "published versionCode $CODE ($NAME) in $UPDATES"
