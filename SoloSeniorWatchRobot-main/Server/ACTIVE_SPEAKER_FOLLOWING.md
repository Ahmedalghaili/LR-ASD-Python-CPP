# Active Speaker Following

The Zenbo Junior turns toward the person who is speaking and looks at them
with its face. When someone else starts speaking, it moves its attention to
the new speaker.

- **Who is speaking** is always decided by our SCRFD + LR-ASD pipeline, so
  the robot follows the actual speaker and not just any face it can see.
- **Where a voice comes from** can also be taken from Zenbo's own voice
  detection (DOA, direction of arrival). This only turns the robot toward a
  speaker the camera cannot see; LR-ASD then confirms who it is.
- **Movement** is rotation only: body turns, head tilt and face expressions.
  The robot never drives forward or backward.

## ASUS built-in functions: what is reused and why

These were checked against the PyZenbo 1.0.46 documentation, the Java
`ZenboJuniorSDK.jar` that this app uses, and svenschultze/Zenbo-LLM-Agent.

| Built-in (PyZenbo → Java) | Decision |
|---|---|
| `motion.move_body(x, y, relative_theta_degree, speed_level)` → `moveBody(x, y, int, SpeedLevel.Body)` | **Used** for all body turns (x = y = 0). An optional speed level (L1–L7) gives smoother turns. |
| `motion.move_head(yaw, pitch)` → `moveHead` | **Used for pitch only.** The docs say yaw has "no effect on Zenbo Junior", and pitch is limited to −10…50°. The body therefore does all horizontal turning. |
| `utility.look_at_user(doa)` → `lookAtUser(float)` | **Used (optional)** to turn toward an unseen speaker. It takes Zenbo's own DOA value, so no unit or sign calibration is needed. |
| `RobotCallback.Listen.onVoiceDetect` / `onEventUserUtterance` | **Used** as the DOA source. Their JSON carries `doa`, `sound_level` and `vad_status` (fields confirmed in the SDK jar). |
| `dialog_system.wait_for_doa(sentence)` | **Not used.** It blocks until Zenbo finishes a speak-and-listen dialogue and returns one DOA, so it cannot monitor a conversation continuously. |
| `utility.track_face()` / `follow_face()` | **Not used.** They follow *a* face with no idea who is speaking, `follow_face` also drives the robot, and both run Zenbo's own vision on the camera this app is already streaming. When another client takes the camera, the app's `onDisconnected` closes it and never reopens it, which would stop the server's video. |
| `vision_control.request_detect_face()` | **Not used**, for the same camera reason. SCRFD already does this on the server. |

Zenbo-LLM-Agent uses `onVoiceDetect` only for the trigger word and head button
(`vad_status == "BeginCSR_TriggerWord"`), and runs its own VAD for speech. It
has no DOA-based following to reuse. So it is still unconfirmed whether Zenbo
reports DOA continuously during normal conversation while this app records
the microphone. That is why DOA starts in `log` mode (see below).

## Using it

Choose a **Move mode** in the server window:

| Move mode | Behaviour |
|---|---|
| Manual | Unchanged. The robot only moves when you command it. |
| Move body / Move head | Unchanged pose-based tracking. |
| **Follow speaker (simulation)** | Makes every decision, logs it and draws it on the preview. **Sends nothing to the robot.** It keeps a virtual body yaw and head pitch. |
| **Follow speaker (robot)** | Sends the decisions to the Zenbo. |

Start without the UI:

```bash
ACTIVE_SPEAKER_FOLLOW=simulation LR_ASD_DETECTOR=scrfd LR_ASD_CROP_ALL_FRAMES=1 \
  ./run_with_lrasd.sh build/SoloSeniorWatchRobot --SettingFile json/ROGG16_SSWR.json
```

| Variable | Values | Meaning |
|---|---|---|
| `ACTIVE_SPEAKER_FOLLOW` | `simulation`, `robot` | Start in that Move mode. |
| `ACTIVE_SPEAKER_DOA` | `off`, **`log`** (default), `look_at_user`, `rotate` | How Zenbo's DOA is used (see below). |
| `ACTIVE_SPEAKER_DOA_SIGN`, `ACTIVE_SPEAKER_DOA_OFFSET` | `1`/`-1`, degrees | Converts raw DOA to a body angle (+ = left). Needed for `rotate`; also used to decide whether a voice comes from inside the camera view. |
| `ACTIVE_SPEAKER_BODY_SPEED` | `1`–`7` | Zenbo `moveBody` speed level. Unset keeps the app default. |
| `LR_ASD_CROP_ALL_FRAMES` | `1` | **Recommended.** Every frame gets its own face crop instead of one crop repeated five times. In the live test this raised speaking detections per person from 5–11 to 25–29 and made speaker switches work. LR-ASD costs 21–33 ms per 200 ms update instead of 11 ms, on its own thread. |
| `SSWR_LISTEN_ADDRESS` | e.g. `127.0.0.1` | Listen on that address only. Use it when testing with the fake robot, so the real Zenbo cannot connect. |

### DOA modes

- **`log`** (default): every Zenbo voice event is logged as
  `{"component":"ZenboVoice",...,"doa":N}`. When exactly one visible face is
  speaking, the follower also logs `voice_speaker_bearing`, which gives
  (raw DOA, true bearing) pairs for calibration. The robot never moves
  because of DOA.
- **`look_at_user`**: turns with Zenbo's built-in `lookAtUser(doa)`.
- **`rotate`**: turns the body by `sign × doa + offset` degrees. Use it only
  after calibrating from the `log` pairs.

A voice turn happens only when all of these hold:

- no visible face has been marked speaking by LR-ASD for 0.8 s;
- the voice comes from outside the camera view (|calibrated DOA| ≥ 25°);
- no visible face lies within 15° of the voice direction;
- at least 1.5 s have passed since following started (the warm-up);
- at least 4 s have passed since the last voice turn.

After a voice turn, a face near the centre is attended as soon as LR-ASD marks
it speaking (no 400 ms wait). Outside that window, the audio direction alone
never selects a face.

## Behaviour and stability

| Situation | Command | Zenbo SDK call |
|---|---|---|
| Speaker off-centre | face AWARE_LEFT / AWARE_RIGHT (eyes glance first) | `robot.setExpression` |
| Speaker centred | face INTERESTED | `robot.setExpression` |
| Speaker outside the 10° dead band | body `degree` | `motion.moveBody(0, 0, degree[, speed])` |
| Face high or low | head pitch (yaw 0) | `motion.moveHead` |
| Unseen speaker (DOA mode) | `look_at_user_doa`, or body `degree` | `utility.lookAtUser` / `motion.moveBody` |
| Speaker lost for 3 s | face DEFAULT, and a body turn back to where they were last seen (if > 10°, at most 90°) | `robot.setExpression`, `motion.moveBody` |

The stability rules are in `ActiveSpeakerConfig` (`ActiveSpeakerFollower.hpp`):

- **Acquire and switch:** a person must speak for 400 ms before the robot
  attends to them, and 600 ms to take attention from another speaker.
- **Dwell:** at least 1.5 s on a speaker.
- **Dead bands:** 10° for the body, 6° for the head.
- **Settle:** no new decision while a turn is in progress.
- **Limits:** at most one turn per second, head steps of at most 8°.
- **Re-identification:** after a turn, a new track where the target should be
  is treated as the same person.
- **Keep the group in view:** when others are visible, the body turns only as
  far as keeps them in the camera view, and the eyes cover the rest.
- **Gaze hysteresis:** the face does not flicker between expressions.
- **Blocked turns:** if two turns in a row barely move the target in the
  image (an obstacle, or a rejected command), rotation pauses for 4 s.
- **Return when lost:** the follower keeps an estimate of the body yaw
  (sum of the turns it sent). When the target is released it turns once
  toward the direction where the target was last seen, so the robot does not
  stay facing an empty part of the room (seen live: it ended up facing the
  window after the speaker left the frame).

Pose tracking and the Visual Compass never compete with the follower. When
the server has not moved the head yet, the follower assumes the Zenbo app's
connect pose, `moveHead(0, 30)`.

## Android app changes (ZenboJrII flavour)

- **`ZenboCallback.onVoiceDetect` / `onEventUserUtterance`:** forward `doa`,
  `sound_level` and `vad_status` to the server, and log the raw JSON.
- **`SocketManager`:**
  - `look_at_user_doa` calls `utility.lookAtUser(doa)`;
  - `bodyspeed` with a rotation calls `moveBody(0, 0, degree, SpeedLevel.Body)`.
- **`RobotCommand.proto`** (shared by the server and the app through a
  symlink) gains optional fields: `look_at_user_doa`, `doa`, `sound_level`
  and `vad_status`. Older builds ignore them.

The new Java code was type-checked and run in a harness against the real
`ZenboJuniorSDK.jar` and protobuf-javalite 3.19.1. **The full app has not been
built here** (there is no Android SDK on this machine). The Zenbo (not Junior)
flavour is unchanged.

## Testing

```bash
cmake --build build --target active_speaker_follower_test active_speaker_replay
(cd build && ctest)                                    # 31 unit tests
build/active_speaker_replay LOG.csv W H [--naive]      # replay recorded LR-ASD output
SSWR_LISTEN_ADDRESS=127.0.0.1 ... ./run_with_lrasd.sh ...   # server for a fake robot
python tests/fake_zenbo_robot.py VIDEO [--window 0.4] [--doa-from-csv LOG.csv]
```

The fake robot streams frames and audio like the Zenbo app. It sends mock DOA
while the recorded LR-ASD log says someone is speaking (±8° noise), and it
executes the commands it receives: turning, tilting, and `lookAtUser`, which
it models as a turn by the DOA angle. **Start the server with
`SSWR_LISTEN_ADDRESS=127.0.0.1`**, or the real Zenbo may connect and receive
the commands. This happened once during development: the robot received face
expressions only.

## Results (2026-09-28)

### Full server with the fake Zenbo

Two-person video, 25 FPS input, SCRFD, `LR_ASD_CROP_ALL_FRAMES=1`, localhost
only, with an unrelated training job using the GPU.

| Scenario | Result |
|---|---|
| Wide view, DOA `log` | 4 speaker switches, a face glance toward each, one 5° turn |
| Wide view, DOA `look_at_user` | Identical: 4 switches, **zero voice turns** (visible faces explain the voices) |
| Narrow view (nobody visible at start), DOA `log` | The robot never finds anyone: the active speaker is in view **0%** of speaking time |
| Narrow view, DOA `look_at_user` | Turns toward each out-of-view voice (42–104°); the active speaker is in view **19–47%** of speaking time (2 runs) |
| Simulation mode, DOA `look_at_user` | 5 switches; **no movement commands sent** |

In every run the camera processed 25.0 FPS, equal to the input rate. The
narrow-view gain is limited because speaking turns in this video last about
1 s, while an 85° turn takes about 2 s at the assumed 45°/s.

### Replay of recorded LR-ASD output

Two-person video plus six 60 s AVA clips, camera only:

- **Rotations:** 0.2–2.2 per 10 s, against 4.8–15.0 when turning at every
  LR-ASD change.
- **Attention on the speaker, allowing 1.5 s to react:** 87–100%.

### Not yet tested on the physical Zenbo

- whether `onVoiceDetect` delivers DOA during normal conversation while this
  app records the microphone (check the `ZenboVoice` log lines);
- the DOA convention (use the `log` calibration pairs), and whether
  `lookAtUser` only rotates or also uses the camera;
- the real turn speed (assumed 45°/s) and the direction of the gaze faces
  (`mirrorGaze`).
