package tw.edu.cgu.ai.voicefollow;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;

import com.asus.robotframework.API.MotionControl;
import com.asus.robotframework.API.RobotAPI;
import com.asus.robotframework.API.RobotCallback;
import com.asus.robotframework.API.RobotCmdState;
import com.asus.robotframework.API.RobotCommand;
import com.asus.robotframework.API.RobotErrorCode;
import com.asus.robotframework.API.RobotFace;
import com.asus.robotframework.API.SpeakConfig;

import org.json.JSONObject;

/**
 * Zenbo Voice Follow: runs only on the robot. Zenbo listens (speakAndListen)
 * and, for each voice direction it reports (onDOA, degrees, + = Zenbo's left,
 * measured: left +90, right -90, behind about 170), turns its body toward it
 * with moveBody. Voices from the front (|doa| <= 20) need no turn. While a
 * turn runs, and for a moment after, new directions are ignored.
 * This app records nothing itself, so Zenbo's listening keeps the microphone.
 * Tap the screen to stop/start.
 */
public class MainActivity extends Activity {
    private static final int FRONT_DEG = 20;          // no turn for a voice this close to the front
    private static final long AFTER_TURN_MS = 1500;   // ignore directions this long after a turn

    private RobotAPI api;
    private DirectionView view;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private volatile boolean running = false, listenActive = false, turning = false;
    private volatile long listenStartedMs = 0, turnStartedMs = 0, turnEndedMs = 0;
    private volatile int turnSerial = -1;

    private final Callback callback = new Callback();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        view = new DirectionView(this);
        setContentView(view);
        api = new RobotAPI(this, callback);
        api.robot.registerListenCallback(callback);
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setRunning(!running);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        api.robot.setExpression(RobotFace.HIDEFACE);
        api.motion.moveHead(0, 30, MotionControl.SpeedLevel.Head.L2);   // look straight ahead
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                setRunning(true);
            }
        }, 1500);
    }

    @Override
    protected void onPause() {
        setRunning(false);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        api.release();
        super.onDestroy();
    }

    private void setRunning(boolean on) {
        running = on;
        handler.removeCallbacks(tick);
        if (on) {
            listenActive = false;
            handler.post(tick);
            view.setStatus(true, "Listening: speak from any side, I will turn to you");
        } else {
            view.setStatus(false, "Stopped (tap to start)");
        }
    }

    // One listen at a time; a new one only after the previous finished (never
    // stopSpeakAndListen: interrupting Zenbo made it stop reporting).
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!running)
                return;
            long now = System.currentTimeMillis();
            if (turning && now - turnStartedMs > 8000) {
                turning = false;                       // no end reported: carry on
                turnEndedMs = now;
            }
            if (!listenActive || now - listenStartedMs > 20000) {
                api.robot.speakAndListen("", new SpeakConfig().timeout(10));
                listenActive = true;
                listenStartedMs = now;
            }
            handler.postDelayed(this, 300);
        }
    };

    private void onDirection(final int doa) {
        view.onDoa(doa);
        long now = System.currentTimeMillis();
        if (!running || turning || now - turnEndedMs < AFTER_TURN_MS)
            return;
        int d = ((doa % 360) + 540) % 360 - 180;       // -180..179, + = left
        if (Math.abs(d) <= FRONT_DEG) {
            view.addEvent("voice in front: no turn");
            return;
        }
        turning = true;
        turnStartedMs = now;
        turnSerial = api.motion.moveBody(0f, 0f, d);    // + = turn left
        view.addEvent("turning " + d + " deg toward the voice");
        view.setStatus(true, "Turning " + d + " deg toward " + DirectionViewNames.name(d));
    }

    /** Side names for the status line. */
    private static class DirectionViewNames {
        static String name(int d) {
            return new String[]{"FRONT", "LEFT", "BEHIND", "RIGHT"}[DirectionView.sector(d)];
        }
    }

    private class Callback extends RobotCallback implements RobotCallback.Listen {
        @Override
        public void onStateChange(int cmd, int serial, RobotErrorCode err, RobotCmdState state) {
            boolean done = state == RobotCmdState.SUCCEED || state == RobotCmdState.FAILED
                    || state == RobotCmdState.PREEMPTED || state == RobotCmdState.REJECTED;
            if (cmd == RobotCommand.DS_SPEAK_AND_LISTEN.getValue() && done) {
                listenActive = false;
            } else if (cmd == RobotCommand.MOTION_MOVE_BODY.getValue() && done) {
                turning = false;
                turnEndedMs = System.currentTimeMillis();
                if (state == RobotCmdState.SUCCEED) {
                    view.addEvent("turn done");
                    view.setStatus(true, "Listening: speak from any side, I will turn to you");
                } else {
                    // e.g. REJECTED COMMON_CHARGING_AC: unplug the charger
                    view.addEvent("turn " + state + ": " + err);
                    view.setStatus(false, "Turn refused: " + err + (String.valueOf(err).contains("CHARGING")
                            ? " - unplug the charger" : ""));
                }
            }
        }

        @Override
        public void onVoiceDetect(JSONObject json) {
            Log.d("VoiceFollow", "onVoiceDetect " + json);
            JSONObject e = json == null ? null : json.optJSONObject("event_vad_status");
            if (e == null)
                return;
            String vad = e.optString("vad_status", "");
            if ("onDOA".equals(vad) && e.has("doa"))
                onDirection(e.optInt("doa"));
            else
                view.addEvent(vad);
        }

        @Override
        public void onEventUserUtterance(JSONObject json) {
            listenActive = false;
        }

        @Override
        public void onResult(JSONObject json) {
        }

        @Override
        public void onRetry(JSONObject json) {
        }

        @Override
        public void onFinishRegister() {
        }

        @Override
        public void onSpeakComplete(String s, String s1) {
        }
    }
}
