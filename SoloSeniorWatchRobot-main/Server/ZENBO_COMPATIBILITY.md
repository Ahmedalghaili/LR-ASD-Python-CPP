# Zenbo compatibility report (original ASUS Zenbo)

Robot: original ASUS Zenbo, firmware 13.10.5.129, Android 6.0.1 (API 23).
Branch: `zenbo-original-compat`. Restore point before this work: tag
`save-2026-09-29-follow-speaker`.

## 1. Which app runs on our robot

The Android project has three flavours:

| Flavour | SDK jar | Android needed | Runs on our robot? |
|---|---|---|---|
| **Zenbo** | `ZenboSDK.jar` | 6.0 (minSdk 23) | **Yes** |
| ZenboJrII | `ZenboJuniorSDK.jar` | 10 (minSdk 29) | No, it cannot be installed on Android 6 |
| Kebbi | Nuwa SDK | 9 | No |

So the robot runs the **Zenbo** flavour. The DOA code written earlier went
into ZenboJrII and never ran on this robot. This branch adds it to the
Zenbo flavour.

## 2. What the original Zenbo SDK supports

Checked in the `ZenboSDK.jar` bytecode and the ASUS documentation.

| Capability | In the SDK | Used by our app today | Status |
|---|---|---|---|
| **Body rotation** | `motion.moveBody(0, 0, degrees[, speed L1–L7])` | Yes (`degree` command) | **Works.** Seen on the live robot: + turns left, − turns right |
| **Head yaw (left/right)** | `motion.moveHead(yaw, pitch, speed L1–L3)`; yaw −45…45°, pitch −15…55° | Yes (`yaw`/`pitch` commands) | **Supported.** The sign is unclear (see below) |
| **Head pitch** | same call | Yes | Supported |
| **Sound direction (DOA)** | `onVoiceDetect` / `onEventUserUtterance` JSON carries `doa`, `sound_level`, `vad_status`. Also a private `registerDoaMonitor()` and `enableAlwaysListenMode()` in `ExtraMethod` | Callbacks are registered but **empty**, so DOA is thrown away | **Present in the SDK. Unknown** whether events arrive during normal talk (the app turns the "Hey Zenbo" trigger off) |
| **Turn to a voice** | `utility.lookAtUser(doa)` | No | Present. Behaviour to test |
| **Face tracking** | `utility.trackUser()`, `utility.followUser()`, `vision.requestDetectFace()` | No | Present. **Risk:** uses the same camera as our video stream |

**Head yaw sign conflict:** the ASUS docs say yaw −45 = left and +45 = right.
The existing server code (`Zenbo/LandmarkToRobotAction.cpp`) sends a negative
yaw for a face on the right. One of them is wrong; the PoC `head` step
measures it.

## 3. How the C++ server reaches these functions

No new connection is needed. Everything goes through the existing protobuf
protocol (`RobotCommand.proto`, shared by the server and the app):

| Direction | Port | Fields |
|---|---|---|
| Server → robot | 8896 | `yaw`, `pitch`, `headspeed` (head); `degree`, `bodyspeed` (body); `look_at_user_doa` (lookAtUser); `poc_action` (test-only) |
| Robot → server | 8898 | `RobotToServerMessage` with `description` = `onVoiceDetect` / `onEventUserUtterance`, plus `doa`, `sound_level`, `vad_status` |

The `active-speaker-following` branch already parses the `onVoiceDetect`
messages in `ThreadReceiveMessage` and passes the DOA to the follower.

## 4. Proof of concept (this branch)

**Robot app (Zenbo flavour), 3 files:**
- `ZenboCallback.java`: forwards every voice event with its DOA to the
  server. It can also forward the robot's command states, off by default.
- `SocketManager.java`:
  - `look_at_user_doa` → `utility.lookAtUser`;
  - `bodyspeed` for turns;
  - `poc_action` test actions: `sdk_version`, `states_on/off`,
    `doa_monitor_on/off`, `always_listen_on/off`, `voice_trigger_on/off`,
    `speak_and_listen`, `track_user`, `cancel_all`.
  
  Each action replies `poc_result …`.
- `MainActivity.java`: reports `camera_disconnected` / `camera_error` to the
  server, so interference with the camera becomes visible.

These were type-checked against the real `ZenboSDK.jar` and protobuf-lite.
**The APK has not been built.** This PC has no Android SDK, and adb over
Wi-Fi is off on the robot.

**PC tool:** `tests/zenbo_poc.py`. It is a small stand-alone server that uses
the robot's protocol, so the C++ LR-ASD server is not involved (stop it
first; both use ports 8895–8898). It measures each movement from the camera
itself: the image shift before and after a command gives the real direction,
the size of the turn, and the camera field of view. During every step it
also records camera FPS, gaps between frames, and the audio level.

| Step | Needs the new app? | What it tests |
|---|---|---|
| `observe` | no | camera FPS, audio level, messages |
| `head` | **no** | head yaw 0→±15→±30 and pitch 20/40/30; direction and size |
| `body --allow-body` | **no** | body ±15°, ±30° |
| `voice` | yes | DOA events in 3 modes: default, DOA monitor, always-listen; microphone still streaming? |
| `look` | yes | `lookAtUser(doa)` after each DOA; camera still streaming? |
| `track` | yes | `trackUser()`; does our camera stream stop? |

### Simulation result (fake robot, same protocol)

`zenbo_poc.py observe head body voice --allow-body` against
`fake_zenbo_robot.py --loop`:

| Check | Result |
|---|---|
| Head yaw direction | 8/8 moves measured correctly, 15.0° per 15° step |
| Body direction | 4/4 correct (15.0°, 30.1°) |
| Head pitch | up/down detected correctly |
| Field of view estimate | 62.4° (true value in the simulation: 62.5°) |
| PoC actions | all sent and answered |
| Camera during movement | 24.8–25.2 FPS, largest gap 0.06 s |
| Audio during movement | full stream (10 chunks/s) |

The simulation proves that the tool and the protocol work. The real robot's
directions and field of view still have to be measured on the robot.

## 5. Interference with camera, audio and inference

- **Inference (LR-ASD):** robot commands are a few bytes on the command
  socket. The follower adds no work to the LR-ASD thread. Earlier fake-robot
  runs kept 25.0 FPS with following on.
- **Camera:** our own movement commands do not touch the camera. Zenbo's
  **vision functions** (`trackUser`, `followUser`, `requestDetectFace`, and
  maybe `lookAtUser`) may take the camera. The app's `onDisconnected` never
  reopens it, which would stop our video. The `track` and `look` steps detect
  this through `camera_disconnected` and FPS.
- **Audio:** on Android 6 only one app can record at a time. Zenbo's
  listening (always-listen, speak-and-listen) may take the microphone from
  our `AudioRecord`. The `voice` step shows it as fewer audio chunks per
  second or silence.
- **Camera frame rate** on the live robot is 10–18 FPS. That limit comes
  from the robot and Wi-Fi, not from movement.

## 6. What works, what does not, what is unknown

**Works now (current app, no install):** body rotation, head pitch, head yaw
commands.

**Does not work today:** DOA reaches nobody (empty callbacks), and there is
no `lookAtUser`. Both are fixed in this branch's app code, which is not built.

**Unknown until tested on the robot:**
- the head yaw sign and the real camera field of view (`head` step, runs
  today);
- whether DOA arrives during normal talk and in which mode (`voice` step);
- whether Zenbo's listening takes our microphone;
- whether `lookAtUser` or `trackUser` stop our camera.

## 7. How to run on the robot

1. Stop the LR-ASD server, which uses the same ports.
2. `python tests/zenbo_poc.py observe head`: the head only, safe. Then open
   the Zenbo app as usual (server 192.168.1.54).
3. With free space around the robot:
   `python tests/zenbo_poc.py body --allow-body`
4. After installing the new APK:
   `python tests/zenbo_poc.py voice --seconds 20`, then `look` and `track`.

The head returns to yaw 0 / pitch 30 at the end, or on Ctrl+C. Yaw is limited
to ±30° by default.

## 8. Integration plan for Active Speaker Following

1. Fix the head yaw sign in the server from the `head` measurement. Set the
   follower's field of view from the measured value.
2. **Use the head first, the body second.** The original Zenbo's head can
   turn ±45°. Small corrections become fast, quiet head moves. The body turns
   only when the speaker is beyond the head's range, and the head then
   re-centres. (Zenbo Junior needed body-only turns; the original Zenbo does
   not.)
3. **DOA:** if the `voice` step shows DOA during normal talk, use it to turn
   toward a speaker outside the camera view. Pick the listening mode that
   keeps our microphone streaming. Otherwise, LR-ASD with the camera only.
4. Keep Zenbo's own face tracking **off** if the `track` step shows it
   stopping our camera.
5. Merge into the `active-speaker-following` branch once the robot tests
   pass.
