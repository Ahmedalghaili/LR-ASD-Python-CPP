package tw.edu.cgu.ai.voicetest;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;

import com.asus.robotframework.API.RobotAPI;
import com.asus.robotframework.API.RobotCallback;
import com.asus.robotframework.API.RobotCmdState;
import com.asus.robotframework.API.RobotCommand;
import com.asus.robotframework.API.RobotErrorCode;
import com.asus.robotframework.API.RobotFace;
import com.asus.robotframework.API.SpeakConfig;

import org.json.JSONObject;

/**
 * Zenbo Voice Test: keeps Zenbo listening (speakAndListen) and shows where
 * each voice comes from, using Zenbo's own direction of arrival (onDOA from
 * its 4-microphone speech system). Nothing else runs: no server, camera or
 * microphone recording of this app. Tap the screen to stop/start listening.
 */
public class MainActivity extends Activity {
    private RobotAPI api;
    private DirectionView view;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private volatile boolean listening = false, listenActive = false;
    private volatile long listenStartedMs = 0;

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
                setListening(!listening);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        api.robot.setExpression(RobotFace.HIDEFACE);   // show this screen, not Zenbo's face
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                setListening(true);
            }
        }, 1500);
    }

    @Override
    protected void onPause() {
        setListening(false);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        api.release();
        super.onDestroy();
    }

    private void setListening(boolean on) {
        listening = on;
        handler.removeCallbacks(tick);
        if (on) {
            listenActive = false;
            handler.post(tick);
            view.setStatus(true, "Listening: speak from any side (tap to stop)");
        } else {
            view.setStatus(false, "Stopped (tap to listen)");
        }
    }

    // Start a new listen only after the previous one has finished (never
    // stopSpeakAndListen: interrupting Zenbo made it stop reporting).
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!listening)
                return;
            long now = System.currentTimeMillis();
            if (!listenActive || now - listenStartedMs > 20000) {
                api.robot.speakAndListen("", new SpeakConfig().timeout(10));
                listenActive = true;
                listenStartedMs = now;
                view.addEvent("listen started");
            }
            handler.postDelayed(this, 300);
        }
    };

    private class Callback extends RobotCallback implements RobotCallback.Listen {
        @Override
        public void onStateChange(int cmd, int serial, RobotErrorCode err, RobotCmdState state) {
            if (cmd == RobotCommand.DS_SPEAK_AND_LISTEN.getValue()
                    && (state == RobotCmdState.SUCCEED || state == RobotCmdState.FAILED
                    || state == RobotCmdState.PREEMPTED || state == RobotCmdState.REJECTED)) {
                listenActive = false;
                view.addEvent("listen " + state + (err == null ? "" : " " + err));
            }
        }

        @Override
        public void onVoiceDetect(JSONObject json) {
            Log.d("VoiceTest", "onVoiceDetect " + json);
            JSONObject e = json == null ? null : json.optJSONObject("event_vad_status");
            if (e == null)
                return;
            String vad = e.optString("vad_status", "");
            if ("onDOA".equals(vad) && e.has("doa"))
                view.onDoa(e.optInt("doa"));
            else
                view.addEvent(vad);
        }

        @Override
        public void onEventUserUtterance(JSONObject json) {
            listenActive = false;
            JSONObject e = json == null ? null : json.optJSONObject("event_user_utterance");
            view.addEvent("heard: " + (e == null ? "?" : e.optString("user_utterance", "").replaceAll("[\\[\\]{}\"]", " ").trim()));
        }

        @Override
        public void onResult(JSONObject json) {
        }

        @Override
        public void onRetry(JSONObject json) {
        }

        @Override
        public void onFinishRegister() {
            view.addEvent("listen callback registered");
        }

        @Override
        public void onSpeakComplete(String s, String s1) {
        }
    }
}
