
package tw.edu.cgu.ai;

import android.util.Log;

import com.asus.robotframework.API.RobotCallback;
import com.asus.robotframework.API.RobotCmdState;
import com.asus.robotframework.API.RobotErrorCode;
import com.asus.robotframework.API.RobotCommand;

import com.google.protobuf.Timestamp;
import org.json.JSONObject;

public class ZenboCallback extends RobotCallback implements RobotCallback.Listen {

    public SocketManager socketManager;

    public boolean RobotMovementFinished_Head = true;
    public boolean RobotMovementFinished_Body = true;
    public long TimeStamp_MovementFinished_Head = Long.MIN_VALUE;
    public long TimeStamp_MovementFinished_Body = Long.MIN_VALUE;
    public long TimeStamp_MovementHead_Active = 0;      //can not use Long.Max_Value, out of range after the subtraction.
    public long TimeStamp_MovementBody_Active = 0;
    // Compatibility PoC: when true, every command state change and listen
    // result is forwarded to the server (poc_action "states_on").
    public volatile boolean forwardStates = false;
    // Voice direction test panel on the robot screen (set by MainActivity).
    public VoiceDirectionView directionView;
    // Called when Zenbo's speakAndListen has finished (MainActivity restarts it).
    public Runnable onListenFinished;

    @Override
    public void onStateChange(int cmd, int serial, RobotErrorCode err_code, RobotCmdState state) {
        //check, what will happen is the power core is plugged?
        Log.d("ZenboCallback", "onStateChange " + String.format("cmd %d serial %d ",cmd, serial) + "err_code " + err_code.toString() + " state " + state.toString());
        if (forwardStates)
            sendText("robot_state cmd=" + commandName(cmd) + " serial=" + serial + " state=" + state + " err=" + err_code);
        if (cmd == RobotCommand.DS_SPEAK_AND_LISTEN.getValue()
                && (state == RobotCmdState.SUCCEED || state == RobotCmdState.FAILED || state == RobotCmdState.PREEMPTED)
                && onListenFinished != null)
            onListenFinished.run();
        if (cmd == RobotCommand.MOTION_MOVE_BODY.getValue()) {        //cmd == 39 movebody
            if (state == RobotCmdState.ACTIVE) {
                //It happen this callback is erased by the Zenbo system prompt.
                RobotMovementFinished_Body = false;
                TimeStamp_MovementBody_Active = System.currentTimeMillis();
            } else if (state == RobotCmdState.SUCCEED) {
                RobotMovementFinished_Body = true;
                TimeStamp_MovementFinished_Body = System.currentTimeMillis();
            } else if (state == RobotCmdState.PENDING) {
                //When does it happen? When Zenbo finds an obstacle.
                RobotMovementFinished_Body = false;
            } else if (state == RobotCmdState.REJECTED) {
                //it happens when Zenbo is updating its language dataset.
                //or the power core is plugged, the RobotCmdState.SUCCEED will never come.
                RobotMovementFinished_Body = true;
                TimeStamp_MovementFinished_Body = System.currentTimeMillis();
            } else if (state == RobotCmdState.FAILED){
                //it happens when Zenbo fails to move to the expected location
                RobotMovementFinished_Body = true;
                TimeStamp_MovementFinished_Body = System.currentTimeMillis();
            } else if (state == RobotCmdState.INITIAL) {
                //I never saw this state, maybe this state is language-related.
                RobotMovementFinished_Body = false;
            }else{
                RobotMovementFinished_Body = false;
            }

        }

        if (cmd == RobotCommand.MOVE_HEAD.getValue())        //cmd == 6, moveHead
        {
            //2018/5/28 Chih-Yuan: The callback is unstable. It happens that there is no more callback after ACTIVE
            if (state == RobotCmdState.ACTIVE) {
                RobotMovementFinished_Head = false;
                TimeStamp_MovementHead_Active = System.currentTimeMillis();
            } else if (state == RobotCmdState.SUCCEED) {
                RobotMovementFinished_Head = true;
                TimeStamp_MovementFinished_Head = System.currentTimeMillis();
            } else if (state == RobotCmdState.PENDING) {
                //it happens when Zenbo is waiting the user's permit to move. However, will Zenbo call the onResult() function?
                //2018/5/28: There is a bug. Zenbo won't send another state after sending this state. Thus I have to set this to true;
                RobotMovementFinished_Head = true;
                TimeStamp_MovementFinished_Head = System.currentTimeMillis();
            } else if (state == RobotCmdState.REJECTED) {
                //it happens when Zenbo is updating its language dataset.
                RobotMovementFinished_Head = true;
                TimeStamp_MovementFinished_Head = System.currentTimeMillis();
            } else if (state == RobotCmdState.FAILED){
                //it happens along with an err_code: MOTION_OUT_OF_BOUNDS, if we use an argument out of range
                RobotMovementFinished_Body = true;
                TimeStamp_MovementFinished_Head = System.currentTimeMillis();
            } else {
                TimeStamp_MovementFinished_Head = System.currentTimeMillis();
                RobotMovementFinished_Head = true;
            }
        }

        /*Command Name           Integer Value (getValue())
          SET_EXPRESSION         31
          SPEAK                  33
          STOP_SPEAK             34
          MOTION_MOVE_BODY       39
          SPEAK_FROM_DS          104
         */
        //mRobotAPI.robot.setExpression(face, sentence, config) belongs to RobotCommand.SET_EXPRESSION rather than RobotCommand.SPEAK
        //The first "Zenbo Nurse Helper have the feature of camera recording..." may be the SPEAK_FROM_DS command.
        if( cmd == RobotCommand.SPEAK.getValue() || cmd == RobotCommand.SET_EXPRESSION.getValue() || cmd == RobotCommand.SPEAK_FROM_DS.getValue())
        {
            Log.d("ZenboCallback", "RobotCommand.SPEAK: cmd = " + cmd + " serial = " + serial + " err_code = " + err_code.toString() + " state = " + state.toString());
            if( state == RobotCmdState.SUCCEED)
            {
                //Send a command to Server
                if (socketManager != null && socketManager.mSocketSendMessages != null) {
                    long timestamp = System.currentTimeMillis();
                    Timestamp eventTime = Timestamp.newBuilder()
                            .setSeconds(timestamp / 1000)
                            .setNanos((int) ((timestamp % 1000) * 1000000))
                            .build();

                    RobotCommandOuterClass.RobotToServerMessage message =
                            RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                                    .setEventTime(eventTime)
                                    .setDescription("onTTSComplete")
                                    .build();

                    socketManager.sendAMessage(message, socketManager.mSocketSendMessages);
                }

            }
        }
        super.onStateChange(cmd, serial, err_code, state);
    }

    //
    @Override
    public void onResult(int cmd,
                         int serial,
                         RobotErrorCode err_code,
                         android.os.Bundle result)
    {
        //Check this. If I can catch the result of the speak() function.
        Log.d("ZenboCallback", "onResult: cmd " + Integer.toString(cmd));
        super.onResult(cmd, serial, err_code, result);

    }

    @Override
    public void onFinishRegister() {

    }

    @Override
    public void onVoiceDetect(JSONObject jsonObject) {
        Log.d("ZenboCallback", "onVoiceDetect: " + jsonObject);
        // Show Zenbo's direction on the robot screen (works without the PC).
        if (directionView != null && jsonObject != null) {
            Object vad = findKey(jsonObject, "vad_status", 3);
            Object doa = findKey(jsonObject, "doa", 3);
            directionView.onVoiceEvent(vad instanceof String ? (String) vad : "");
            if ("onDOA".equals(vad) && doa instanceof Number)
                directionView.onDoa(((Number) doa).intValue());
        }
        forwardVoiceEvent("onVoiceDetect", jsonObject);
    }

    // Send a voice event with its direction of arrival (DOA) to the server.
    // Events without a doa are forwarded too, so the server log shows what
    // the robot reports.
    private void forwardVoiceEvent(String description, JSONObject jsonObject) {
        if (socketManager == null || socketManager.mSocketSendMessages == null || jsonObject == null)
            return;
        // The complete raw JSON as well, so the server can see every key
        // Zenbo sends (same as Log.d("ZenboJSON", jsonObject.toString(2))).
        try {
            sendText("zenbo_json " + description + "\n" + jsonObject.toString(2));
        } catch (org.json.JSONException e) {
            sendText("zenbo_json " + description + "\n" + jsonObject);
        }
        RobotCommandOuterClass.RobotToServerMessage.Builder builder =
                RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                        .setEventTime(now())
                        .setDescription(description);
        Object doa = findKey(jsonObject, "doa", 3);
        if (doa instanceof Number)
            builder.setDoa(((Number) doa).intValue());
        Object level = findKey(jsonObject, "sound_level", 3);
        if (level instanceof Number)
            builder.setSoundLevel(((Number) level).intValue());
        Object vad = findKey(jsonObject, "vad_status", 3);
        if (vad instanceof String)
            builder.setVadStatus((String) vad);
        socketManager.sendAMessage(builder.build(), socketManager.mSocketSendMessages);
    }

    // Send a plain text message (PoC results and states) to the server.
    public void sendText(String text) {
        Log.d("ZenboCallback", text);
        if (socketManager == null || socketManager.mSocketSendMessages == null)
            return;
        socketManager.sendAMessage(RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                .setEventTime(now())
                .setDescription(text)
                .build(), socketManager.mSocketSendMessages);
    }

    private static Timestamp now() {
        long timestamp = System.currentTimeMillis();
        return Timestamp.newBuilder()
                .setSeconds(timestamp / 1000)
                .setNanos((int) ((timestamp % 1000) * 1000000))
                .build();
    }

    private static String commandName(int cmd) {
        for (RobotCommand c : RobotCommand.values())
            if (c.getValue() == cmd)
                return c.name();
        return Integer.toString(cmd);
    }

    // Find a key at any nesting level (the event payload may be nested or a
    // JSON string inside the object).
    private static Object findKey(JSONObject json, String key, int depth) {
        if (json == null || depth < 0)
            return null;
        if (json.has(key))
            return json.opt(key);
        java.util.Iterator<String> keys = json.keys();
        while (keys.hasNext()) {
            Object value = json.opt(keys.next());
            JSONObject child = null;
            if (value instanceof JSONObject) {
                child = (JSONObject) value;
            } else if (value instanceof String && ((String) value).startsWith("{")) {
                try {
                    child = new JSONObject((String) value);
                } catch (org.json.JSONException ignored) {
                }
            }
            Object found = findKey(child, key, depth - 1);
            if (found != null)
                return found;
        }
        return null;
    }

    @Override
    public void onSpeakComplete(String s, String s1) {
        Log.d("ZenboCallback", "onSpeakComplete: " + s);
        //Send a command to Server
        if (socketManager != null && socketManager.mSocketSendMessages != null) {
            long timestamp = System.currentTimeMillis();
            Timestamp eventTime = Timestamp.newBuilder()
                    .setSeconds(timestamp / 1000)
                    .setNanos((int) ((timestamp % 1000) * 1000000))
                    .build();

            RobotCommandOuterClass.RobotToServerMessage message =
                    RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                            .setEventTime(eventTime)
                            .setDescription("onTTSComplete")
                            .build();

            socketManager.sendAMessage(message, socketManager.mSocketSendMessages);
            Log.d("ZenboCallback", "Send onSpeakComplete: " + s);
        }
    }

    @Override
    public void onEventUserUtterance(JSONObject jsonObject) {
        Log.d("ZenboCallback", "onEventUserUtterance: " + jsonObject);
        if (onListenFinished != null)
            onListenFinished.run();
        forwardVoiceEvent("onEventUserUtterance", jsonObject);
    }

    @Override
    public void onResult(JSONObject jsonObject) {
        Log.d("ZenboCallback", "onResult: " + jsonObject.toString());
        if (forwardStates)
            forwardVoiceEvent("onListenResult", jsonObject);

    }

    @Override
    public void onRetry(JSONObject jsonObject) {

    }
}
