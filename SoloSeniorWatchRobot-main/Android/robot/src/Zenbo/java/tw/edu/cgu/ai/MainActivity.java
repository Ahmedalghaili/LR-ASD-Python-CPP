/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*  This file has been modified by Nataniel Ruiz affiliated with Wall Lab
 *  at the Georgia Institute of Technology School of Interactive Computing
 */

package tw.edu.cgu.ai;

import static java.lang.Thread.sleep;

import android.Manifest;
import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Toast;

import com.asus.robotframework.API.MotionControl;
import com.asus.robotframework.API.RobotAPI;     //How to release this resource in OnDestroy()
import com.asus.robotframework.API.RobotFace;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import android.widget.Button;

import tw.edu.cgu.ai.env.Logger;

public class MainActivity extends Activity {
    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final String PERMISSION_RECORD_AUDIO = Manifest.permission.RECORD_AUDIO;

    public RobotAPI mRobotAPI;
    private ZenboCallback robotCallback;
    private static final Logger LOGGER = new Logger();

    private InputView inputView;     //2024/6/25 Chih-Yuan Yang: The purpose of the inputView is to get a frame from camera's preview.
    //Thus, I can send the frame to a server.
//    private ActionRunnable mActionRunnable = new ActionRunnable();
    private CheckBox checkBox_enable_connection;
    private EditText editText_Server;
    private EditText editText_Port;
//    private DataBuffer m_DataBuffer;
    //Zenbo supports 1920x1080
    //Emulator only supports up to 1280x960
    private Size mPreviewSize = new Size(640, 480);
    private CameraDevice mCameraDevice;
    private HandlerThread threadImageListener;
    private Handler handlerImageListener;
    private HandlerThread mThreadSendAudio;
    private Handler mHandlerSendAudio;
//    private boolean mbReceiveCommand;

    private ImageReader mPreviewReader;
    private CaptureRequest.Builder mPreviewBuilder;
    private final Semaphore cameraOpenCloseLock = new Semaphore(1);
    private CameraCaptureSession mPreviewSession;
    private final ImageListener mPreviewListener = new ImageListener();
    private final SimpleDateFormat mDateFormat = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss.SSS");

    private SocketManager socketManager = new SocketManager();

    private Converter converter;

    private MediaPlayer mediaPlayer;
    private MediaPlayer mediaPlayer2;

    /**
     * {@link android.view.TextureView.SurfaceTextureListener} handles several lifecycle events on a
     * {@link TextureView}.
     */
    private final TextureView.SurfaceTextureListener surfaceTextureListener =
            new TextureView.SurfaceTextureListener() {
                @Override
                public void  onSurfaceTextureAvailable(
                        final SurfaceTexture texture, final int width, final int height) {
                    if( checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED )
                        openCamera();
                }

                @Override
                public void onSurfaceTextureSizeChanged(
                        final SurfaceTexture texture, final int width, final int height) {
                }

                @Override
                public boolean onSurfaceTextureDestroyed(final SurfaceTexture texture) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(final SurfaceTexture texture) {
                }
            };


    /**
     * {@link android.hardware.camera2.CameraDevice.StateCallback}
     * is called when {@link CameraDevice} changes its state.
     */
    private final CameraDevice.StateCallback mStateCallback =
            new CameraDevice.StateCallback() {
                @Override
                public void onOpened(final CameraDevice cameraDevice) {
                    // This method is called when the camera is opened.  We start camera preview here.
                    mCameraDevice = cameraDevice;
                    startPreview();
                    cameraOpenCloseLock.release();  //Chih-Yuan Yang 2024/6/16: The cameraOpenCloseLock is a semaphore.
                }

                @Override
                public void onDisconnected(final CameraDevice cd) {
                    // Another client (e.g. Zenbo's own vision) took the camera.
                    if (robotCallback != null)
                        robotCallback.sendText("camera_disconnected");
                    cameraOpenCloseLock.release();
                    cd.close();    //2025/1/6 This function is not called.
                    mCameraDevice = null;
                }

                @Override
                public void onError(final CameraDevice cd, final int error) {
                    if (robotCallback != null)
                        robotCallback.sendText("camera_error " + error);
                    cameraOpenCloseLock.release();
                    cd.close();
                    mCameraDevice = null;
                }
            };

    private void showToast(final String text) {
        MainActivity.this.runOnUiThread(
            new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, text, Toast.LENGTH_SHORT).show();
                }
            }
        );
    }

    AudioRecord recorder;

    private int sampleRate = 16000 ; // 44100 for music , 16000 for whisper.cpp
    private int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
    int minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);   //minBufSize = 5376, but larger is better.
    private VoiceDirectionView directionView;
    private volatile boolean doaListening = false, listenActive = false;
    private volatile long listenStartedMs = 0;
    private long listenRegisteredMs = 0;
    private final Handler listenHandler = new Handler(android.os.Looper.getMainLooper());
    // Keep Zenbo listening while the panel's listening is on, so it reports the
    // direction of each voice. Never stopSpeakAndListen: after that Zenbo
    // stopped sending voice events until the app was restarted.
    private final Runnable listenTick = new Runnable() {
        @Override
        public void run() {
            if (!doaListening)
                return;
            long now = System.currentTimeMillis();
            if (!listenActive || now - listenStartedMs > 20000) {
                // Zenbo drops the listen callback after a while (listens still
                // finished, but no voice event reached the app): register it
                // again, at most every 30 s.
                if (now - listenRegisteredMs > 30000) {
                    mRobotAPI.robot.registerListenCallback(robotCallback);
                    listenRegisteredMs = now;
                }
                mRobotAPI.robot.speakAndListen("", new com.asus.robotframework.API.SpeakConfig().timeout(10));
                listenActive = true;
                listenStartedMs = now;
                directionView.setListening(true, "Listening: speak from any side (tap to stop)");
            }
            listenHandler.postDelayed(this, 300);
        }
    };

    private void setDoaListening(boolean on) {
        doaListening = on;
        if (on) {
            listenActive = false;
            listenHandler.post(listenTick);
        } else {
            listenHandler.removeCallbacks(listenTick);
            directionView.setListening(false, "Tap here to start listening");
        }
    }

    private static double dbfs(double rms) {
        return 20 * Math.log10(Math.max(rms, 1.0) / 32768.0);
    }

    // 2 when the recorder records Zenbo's two microphones (then the server can
    // compute the direction of a voice), 1 for the old mono recording.
    private volatile int recorderChannels = 1;

    // Record the two microphones (channel index 0 and 1; the others are silent,
    // see Server/tests/zenbo_mic_probe.py). Fall back to mono if refused.
    private AudioRecord createRecorder() {
        try {
            AudioFormat format = new AudioFormat.Builder()
                    .setEncoding(audioFormat)
                    .setSampleRate(sampleRate)
                    .setChannelIndexMask(0x3)
                    .build();
            AudioRecord r = new AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.MIC)
                    .setAudioFormat(format)
                    .setBufferSizeInBytes(minBufSize * 20)
                    .build();
            if (r.getState() == AudioRecord.STATE_INITIALIZED && r.getChannelCount() == 2) {
                recorderChannels = 2;
                return r;
            }
            r.release();
        } catch (Exception e) {
            Log.e("recorder", "stereo recorder failed: " + e.getMessage());
        }
        recorderChannels = 1;
        return new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, minBufSize * 10);
    }
    private boolean status = true;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        // Catch the setScreenAvoidRMode NoSuchMethodError from Zenbo SDK
        final Thread.UncaughtExceptionHandler defaultHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(Thread t, Throwable e) {
                if (e instanceof NoSuchMethodError && e.getMessage() != null && e.getMessage().contains("setScreenAvoidRMode")) {
                    Log.e("ZenboSDKFix", "Caught and ignored expected NoSuchMethodError: " + e.getMessage());
                } else if (defaultHandler != null) {
                    defaultHandler.uncaughtException(t, e);
                }
            }
        });
        super.onCreate(savedInstanceState);

        setContentView(R.layout.main_activity);

        if(!hasPermission()) {
            //This is a new thread, we don't know when users will finish it.
            requestPermission();
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        robotCallback = new ZenboCallback();
        robotCallback.socketManager = socketManager;
        socketManager.robotCallback = robotCallback;
        socketManager.context = getApplicationContext();
        // Voice direction test panel (bottom right): Zenbo's own direction and
        // the two microphone channels this app records. Tap it to listen.
        directionView = new VoiceDirectionView(this);
        android.widget.FrameLayout.LayoutParams panel = new android.widget.FrameLayout.LayoutParams(
                560, 640, android.view.Gravity.BOTTOM | android.view.Gravity.END);
        panel.setMargins(0, 0, 20, 20);
        addContentView(directionView, panel);
        robotCallback.directionView = directionView;
        robotCallback.onListenFinished = new Runnable() {
            @Override
            public void run() {
                listenActive = false;
            }
        };
        socketManager.onDoaListenOn = new Runnable() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (!doaListening)
                            setDoaListening(true);
                    }
                });
            }
        };
        socketManager.onDoaListenOff = new Runnable() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        setDoaListening(false);
                    }
                });
            }
        };
        directionView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setDoaListening(!doaListening);
            }
        });
        socketManager.onCheckUpdate = new Runnable() {
            @Override
            public void run() {
                AppUpdater.check(MainActivity.this, socketManager.mServerURL, robotCallback);
            }
        };
        mRobotAPI = new RobotAPI(this, robotCallback);
        mRobotAPI.robot.registerListenCallback(robotCallback);
        socketManager.mRobotAPI = mRobotAPI;
        socketManager.startThreads();
        mediaPlayer = MediaPlayer.create(this, R.raw.zenbo_love_you_too_manderin); // R.raw.your_mp3_filename
        mediaPlayer2 = MediaPlayer.create(this, R.raw.chipi_chipi_chapa_chapa); // R.raw.your_mp3_filename
        socketManager.mediaPlayer = mediaPlayer;
        socketManager.mediaPlayer2 = mediaPlayer2;

        inputView = (InputView) findViewById(R.id.inputview);
        editText_Port = (EditText) findViewById(R.id.editText_Port);
        editText_Server = (EditText) findViewById(R.id.editText_Server);
        checkBox_enable_connection = (CheckBox) findViewById(R.id.checkBox_connect);
        Button button_close = (Button) findViewById(R.id.button_close);

        //get the default ServerURL
        SharedPreferences sharedPref = getSharedPreferences("ZenboNurseHelper_Preference", Context.MODE_PRIVATE);
        String ServerURL = sharedPref.getString("ServerURL", "");
        if( !ServerURL.isEmpty() ){
            editText_Server.setText(ServerURL);
        }

        checkBox_enable_connection.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    //Save the IP address to SharedPreferences
                    SharedPreferences sharedPref = getSharedPreferences("ZenboNurseHelper_Preference", Context.MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPref.edit();
                    editor.putString("ServerURL", editText_Server.getText().toString());
                    editor.apply();

                    if (recorder != null && recorder.getState() == AudioRecord.STATE_INITIALIZED)
                        recorder.startRecording();    //The recorder means the audio recorder
                    else
                        Toast.makeText(MainActivity.this, "Microphone permission missing: no audio", Toast.LENGTH_LONG).show();
                    socketManager.mServerURL = editText_Server.getText().toString();
                    socketManager.mPortNumber = Integer.parseInt(editText_Port.getText().toString());
                    Log.d("Record", "Before call connectSockets");
                    socketManager.connectSockets();
                    // A newer version of this app on the PC? Then offer to install it.
                    AppUpdater.check(MainActivity.this, socketManager.mServerURL, robotCallback);
                    socketManager.startReceiveCommands();
                    socketManager.startDisconnectionChecker();

                    //How can I know it is connected? no way. I need a loop
                    int CheckCount = 0;
                    while(true) {
                        try {
                            sleep(500);
                        } catch (Exception e) {
                            Log.d("Exception", e.getMessage());
                        }
                        if (socketManager.isConnected()) {
                            mRobotAPI.motion.moveHead(0, 30, MotionControl.SpeedLevel.Head.L3);
                            break;
                        }
                        CheckCount++;
                        if(CheckCount > 10)
                            break;
                    }
                }
                else {
                    //2025/1/3 the recorder should stop in onPause()
//                    mbReceiveCommand = false;   //the app should not mSocketReceiveResults.getInputStream()
                    if (recorder != null && recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING)
                        recorder.stop();
                    socketManager.stopReceiveCommands();
                    socketManager.disconnectSockets();
                }
            }
        });

        button_close.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                finish();
                                            }
                                        }
        );

        mRobotAPI.robot.setExpression(RobotFace.HIDEFACE);
        mRobotAPI.robot.setPressOnHeadAction(false);
        mRobotAPI.robot.setVoiceTrigger(false);     //disable the voice trigger

        mRobotAPI.robot.speak("哈囉，你好。");
    }  //end of onCreate

    //2025/1/3 This is a call back function, using the same thread as onCreate(). Thus, it is only be called after the onCreated is completed.
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED
                        && grantResults[2] == PackageManager.PERMISSION_GRANTED) {
                    startThreads();
                } else {
                    requestPermission();
                }
            }
        }
    }


    @Override
    protected void onResume() {
        super.onResume();
        socketManager.startThreads();
        startThreads();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_STABLE;
        decorView.setSystemUiVisibility(uiOptions);

        // A recorder made before RECORD_AUDIO was granted (first start of a
        // fresh install) never works; make a new one once permission exists.
        if (recorder != null && recorder.getState() != AudioRecord.STATE_INITIALIZED) {
            recorder.release();
            recorder = null;
        }
        if(recorder == null && checkSelfPermission(PERMISSION_RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            recorder = createRecorder();
            Log.d("VS", "Recorder initialized");
        }
        if( checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED )
            openCamera();

        StartAudioRecorder();
    }

    private void StartAudioRecorder()
    {
        //Start audio recorder
        mHandlerSendAudio.post(
            new Runnable() {
                @Override
                public void run() {
                    short[] buffer = new short[minBufSize * 2];   //minBufSize = 5376; x2 for stereo
                    short[] stereoChunk = new short[sampleRate / 5 * 2];   // 200 ms of both microphones
                    int stereoFill = 0;
                    Log.d("VS", "Buffer created of size " + minBufSize);           // every 5 second, the log message occurs. It does not make sense.

                    int readSize;
                    int okReads = 0, errorReads = 0, lastError = 0;
                    long lastReport = 0, lastStartTry = 0;
                    while (status) {
                        long now = System.currentTimeMillis();
                        boolean connected = socketManager.isConnected();
                        // Test switch (poc_action mic_off / mic_on): free the microphone completely.
                        if (!socketManager.micEnabled && recorder != null) {
                            try {
                                if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING)
                                    recorder.stop();
                            } catch (Exception ignored) {
                            }
                            recorder.release();
                            recorder = null;
                        }
                        if (socketManager.micEnabled && recorder == null
                                && checkSelfPermission(PERMISSION_RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                            recorder = createRecorder();
                        }
                        // A recorder made while the microphone was busy never works: make a new one.
                        if (recorder != null && recorder.getState() != AudioRecord.STATE_INITIALIZED
                                && now - lastStartTry > 5000) {
                            lastStartTry = now;
                            recorder.release();
                            recorder = createRecorder();
                        }
                        AudioRecord r = recorder;
                        // Record whenever the app is connected, also after an automatic reconnect.
                        if (r != null && connected && r.getState() == AudioRecord.STATE_INITIALIZED
                                && r.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING
                                && now - lastStartTry > 3000) {
                            lastStartTry = now;
                            try {
                                r.startRecording();
                            } catch (Exception e) {
                                lastError = -1000;
                            }
                        }
                        // Tell the server every 5 s what the microphone is doing.
                        if (connected && now - lastReport > 5000 && robotCallback != null) {
                            lastReport = now;
                            robotCallback.sendText("audio_status mic=" + (socketManager.micEnabled ? "on" : "OFF(test)")
                                    + " channels=" + (r == null ? 0 : r.getChannelCount())
                                    + " permission="
                                    + (checkSelfPermission(PERMISSION_RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED ? "yes" : "NO")
                                    + " recorder=" + (r == null ? "none" : r.getState() == AudioRecord.STATE_INITIALIZED ? "ready" : "FAILED")
                                    + " recording=" + (r != null && r.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING ? "yes" : "NO")
                                    + " reads_ok=" + okReads + " read_errors=" + errorReads + " last_error=" + lastError);
                            okReads = 0;
                            errorReads = 0;
                        }
                        if (r == null || r.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
                            try { sleep(200); } catch (Exception e) { return; }
                            continue;
                        }
                        //reading data from MIC into buffer
                        int channels = r.getChannelCount();
                        readSize = r.read(buffer, 0, channels == 2 ? buffer.length : minBufSize);
                        if( readSize >= 0 && channels == 2) {
                            okReads++;
                            // Mono for the server's speech pipeline, exactly as before:
                            // the average of the two microphones.
                            int frames = readSize / 2;
                            short[] mono = new short[frames];
                            for (int i = 0; i < frames; i++)
                                mono[i] = (short) ((buffer[2 * i] + buffer[2 * i + 1]) / 2);
                            socketManager.sendAudio(converter.ShortToByte_Twiddle_Method(mono));
                            if (directionView != null && frames > 0) {
                                // Levels of both channels, and whether they differ
                                // (two microphones) or are one signal copied twice.
                                double s0 = 0, s1 = 0, sd = 0;
                                for (int i = 0; i < frames; i++) {
                                    double a = buffer[2 * i], b = buffer[2 * i + 1];
                                    s0 += a * a;
                                    s1 += b * b;
                                    sd += (a - b) * (a - b);
                                }
                                double r0 = Math.sqrt(s0 / frames), r1 = Math.sqrt(s1 / frames), rd = Math.sqrt(sd / frames);
                                directionView.setChannels(dbfs(r0), dbfs(r1),
                                        rd < 0.01 * Math.max(Math.max(r0, r1), 1.0) ? "SAME (one signal copied)" : "DIFFERENT");
                            }
                            // Both microphones in 200 ms pieces for the voice direction.
                            for (int i = 0; i < frames * 2; i++) {
                                stereoChunk[stereoFill++] = buffer[i];
                                if (stereoFill == stereoChunk.length) {
                                    socketManager.sendStereoAudio(converter.ShortToByte_Twiddle_Method(stereoChunk), sampleRate);
                                    stereoFill = 0;
                                }
                            }
                        }
                        else if( readSize >= 0) {
                            okReads++;
                            short[] mono = java.util.Arrays.copyOf(buffer, minBufSize);
                            byte[] byteBuffer = converter.ShortToByte_Twiddle_Method(mono);
                            socketManager.sendAudio(byteBuffer);
                            if (directionView != null && readSize > 0) {
                                double s0 = 0;
                                for (int i = 0; i < readSize; i++)
                                    s0 += (double) buffer[i] * buffer[i];
                                directionView.setChannels(dbfs(Math.sqrt(s0 / readSize)), -120, "MONO (1 channel)");
                            }
                        }
                        else
                        {
                            errorReads++;
                            lastError = readSize;
                            Log.e("recorder","recorder.read() error " + readSize);
                            try { sleep(50); } catch (Exception e) { return; }
                        }
                    }
                }
            }
        );
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE) ||
                    shouldShowRequestPermissionRationale(PERMISSION_RECORD_AUDIO)) {
                Toast.makeText(this, "Please grant permissions to execute this program", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA, PERMISSION_STORAGE, PERMISSION_RECORD_AUDIO}, PERMISSIONS_REQUEST);
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        socketManager.disconnectSockets();
        status = false;
        if( recorder != null) {
            try {
                if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop();
                }
                recorder.release();
            } catch (Exception e) {
                Log.e("MainActivity", "Error releasing recorder: " + e.getMessage());
            }
            recorder = null;
        }
        Log.d("VS","Recorder released");

        super.onPause();
    }

    @Override
    protected void onStop()
    {
        super.onStop();
        stopThreads();      //Which is better? onPause() or onStop()?
        socketManager.stopThreads();
    }

    @Override
    protected void onDestroy()
    {
        if( mRobotAPI != null)
            mRobotAPI.release();
        super.onDestroy();
    }

//    /**
//     * Opens the camera specified by {@link CameraConnectionFragment#cameraId}.
//     */
    @SuppressLint("MissingPermission")
    @TargetApi(Build.VERSION_CODES.M)
    private void openCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!cameraOpenCloseLock.tryAcquire(30000, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            String cameraId = manager.getCameraIdList()[0];
            manager.openCamera(cameraId, mStateCallback, handlerImageListener);
        } catch (final CameraAccessException e) {
            LOGGER.e(e, "Exception!");
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }

    /*
    2025/01/06 Chih-Yuan: This is an utility function. I did not use it for ZenboNurseHelper.
     */
    private void listCameraSupportedFormats() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!cameraOpenCloseLock.tryAcquire(30000, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            String cameraId = manager.getCameraIdList()[0];    //Chih-Yuan Yang 2024/6/16: Use the first camera, and Zenbo has only 1 camera
            // Choose the sizes for camera preview and video recording
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

            if (map == null) {
                throw new RuntimeException("Cannot get available preview/video sizes");
            }
            else {
                int format_list[] = map.getOutputFormats();
                for( int format : format_list )
                {
                    Log.d("format",String.format("Support format %d", format));
                    Size Sizes[] = map.getOutputSizes(format);
                    for( Size size : Sizes)
                    {
                        Log.d("OutputSizes",String.format("width %d height %d", size.getWidth(), size.getHeight()));
                    }
                }
            }

        } catch (final CameraAccessException e) {
            LOGGER.e(e, "Exception!");
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }


    /**
     * Closes the current {@link CameraDevice}.
     */
    private void closeCamera() {
        try {
            cameraOpenCloseLock.acquire();
            if (null != mPreviewSession) {
                mPreviewSession.close();
                mPreviewSession = null;
            }
            if (null != mCameraDevice) {
//                mCameraDevice.close();      //2025/1/6 Here is an error message. Do I not need to close the camera again?
                //When does the mCameraDevice object close?
                //the mCameraDevice is close in the backback class's onDisconnected()
                mCameraDevice = null;
            }
            if (null != mPreviewReader) {
                mPreviewReader.close();
                mPreviewReader = null;
            }
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            cameraOpenCloseLock.release();
        }
    }

    /**
     * Starts a background thread and its {@link Handler}.
     */
    private void startThreads() {
        threadImageListener = new HandlerThread("ImageListener");
        threadImageListener.start();
        handlerImageListener = new Handler(threadImageListener.getLooper());

        mThreadSendAudio = new HandlerThread(("threadSendAudio"));
        mThreadSendAudio.start();
        mHandlerSendAudio = new Handler(mThreadSendAudio.getLooper());
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopThreads() {
        threadImageListener.quitSafely();
        mThreadSendAudio.quitSafely();

        try {
            threadImageListener.join();
            threadImageListener = null;
            handlerImageListener = null;

            mThreadSendAudio.join();
            mThreadSendAudio = null;
            mHandlerSendAudio = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }
    }


    private void closePreviewSession() {
        if (mPreviewSession != null) {
            mPreviewSession.close();
            mPreviewSession = null;
        }
    }

    /**
     * Update the camera preview. {@link #startPreview()} needs to be called in advance.
     */
    private void updatePreview() {
        if (null == mCameraDevice) {
            return;
        }
        try {
            //CaptureRequest.CONTROL_MODE: Overall mode of 3A
            mPreviewBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, handlerImageListener);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Start the camera preview.
     */
    private void startPreview() {
        if (null == mCameraDevice || null == mPreviewSize) {
            return;
        }
        try {
            closePreviewSession();
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            mPreviewReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.YUV_420_888, 2);
//            mPreviewReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.JPEG, 2);
            mPreviewReader.setOnImageAvailableListener(mPreviewListener, handlerImageListener);
            mPreviewBuilder.addTarget(mPreviewReader.getSurface());

            mCameraDevice.createCaptureSession(
                   Arrays.asList(mPreviewReader.getSurface()),
                   new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            mPreviewSession = session;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            showToast("Failed");
                        }
                    }, handlerImageListener);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        mPreviewListener.initialize(socketManager, inputView);
    }

}
