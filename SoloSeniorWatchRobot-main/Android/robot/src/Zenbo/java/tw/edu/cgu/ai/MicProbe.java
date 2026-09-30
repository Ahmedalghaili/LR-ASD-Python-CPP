package tw.edu.cgu.ai;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;

/**
 * Microphone probe for the direction-of-arrival test (Server/tests/zenbo_mic_probe.py).
 *
 * The app normally records one mono channel, which cannot show where a voice
 * comes from. This class tries other audio sources and multi-channel setups,
 * records a short clip with each, and sends the raw interleaved PCM16 audio
 * to the server, which checks whether the channels are different microphones.
 * The normal recording is paused (SocketManager.micEnabled) while it runs,
 * because Android 6 lets only one recorder use the microphone.
 */
class MicProbe {
    private static final int[] SOURCES = {
            MediaRecorder.AudioSource.MIC,
            MediaRecorder.AudioSource.CAMCORDER,
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            MediaRecorder.AudioSource.DEFAULT};
    private static final String[] SOURCE_NAMES = {
            "MIC", "CAMCORDER", "VOICE_RECOGNITION", "VOICE_COMMUNICATION", "DEFAULT"};
    // "stereo" = CHANNEL_IN_STEREO; "indexN" = the first N input channels by index.
    private static final String[] SETUPS = {"stereo", "index2", "index4", "index6", "index8"};
    private static final int[] RATES = {48000, 16000};

    private final SocketManager socketManager;
    private volatile boolean busy = false;

    MicProbe(SocketManager socketManager) {
        this.socketManager = socketManager;
    }

    /** "mic_probe": every source x setup x rate, 1 s each. */
    void probeAll() {
        runExclusive(new Runnable() {
            @Override
            public void run() {
                for (int rate : RATES)
                    for (int s = 0; s < SOURCES.length; s++)
                        for (String setup : SETUPS)
                            record("mic_probe", s, setup, rate, 1.0f);
                send("mic_probe_done", null, 0, 0);
            }
        });
    }

    /** "mic_record:SOURCE:SETUP:RATE:SECONDS", e.g. mic_record:MIC:index4:48000:4 */
    void recordOne(String spec) {
        final String[] p = spec.split(":");
        runExclusive(new Runnable() {
            @Override
            public void run() {
                try {
                    int s = java.util.Arrays.asList(SOURCE_NAMES).indexOf(p[1]);
                    record("mic_record", s < 0 ? 0 : s, p[2], Integer.parseInt(p[3]), Float.parseFloat(p[4]));
                } catch (Exception e) {
                    send("mic_record result=bad request " + spec, null, 0, 0);
                }
            }
        });
    }

    private void runExclusive(final Runnable job) {
        if (busy) {
            send("mic_probe result=busy", null, 0, 0);
            return;
        }
        busy = true;
        new Thread(new Runnable() {
            @Override
            public void run() {
                boolean wasEnabled = socketManager.micEnabled;
                socketManager.micEnabled = false;   // free the microphone
                try {
                    Thread.sleep(1000);
                    job.run();
                } catch (Throwable e) {
                    Log.e("MicProbe", "probe failed", e);
                } finally {
                    socketManager.micEnabled = wasEnabled;
                    busy = false;
                }
            }
        }, "MicProbe").start();
    }

    private void record(String tag, int sourceIndex, String setup, int rate, float seconds) {
        AudioRecord recorder = null;
        String result;
        int channels = 0;
        int actualRate = rate;
        byte[] pcm = null;
        try {
            AudioFormat.Builder fb = new AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(rate);
            if (setup.equals("stereo"))
                fb.setChannelMask(AudioFormat.CHANNEL_IN_STEREO);
            else
                fb.setChannelIndexMask((1 << Integer.parseInt(setup.substring(5))) - 1);
            AudioFormat format = fb.build();
            int wanted = Math.max(format.getChannelCount(), 1);
            recorder = new AudioRecord.Builder()
                    .setAudioSource(SOURCES[sourceIndex])
                    .setAudioFormat(format)
                    .setBufferSizeInBytes(rate * wanted * 2)
                    .build();
            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                result = "init_failed";
            } else {
                channels = recorder.getChannelCount();
                actualRate = recorder.getSampleRate();
                recorder.startRecording();
                if (recorder.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
                    result = "start_failed";
                } else {
                    short[] samples = new short[(int) (actualRate * seconds) * channels];
                    int got = 0;
                    while (got < samples.length) {
                        int n = recorder.read(samples, got, Math.min(4096, samples.length - got));
                        if (n <= 0)
                            break;
                        got += n;
                    }
                    pcm = new byte[got * 2];
                    for (int i = 0; i < got; i++) {
                        pcm[2 * i] = (byte) (samples[i] & 0xff);
                        pcm[2 * i + 1] = (byte) ((samples[i] >> 8) & 0xff);
                    }
                    result = got == samples.length ? "ok" : "short_read " + got;
                }
            }
        } catch (Throwable e) {
            result = "error " + e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            if (recorder != null) {
                try {
                    recorder.stop();
                } catch (Throwable ignored) {
                }
                recorder.release();
            }
        }
        send(tag + " source=" + SOURCE_NAMES[sourceIndex] + " setup=" + setup + " channels=" + channels
                + " rate=" + actualRate + " result=" + result, pcm, channels, actualRate);
    }

    private void send(String description, byte[] pcm, int channels, int rate) {
        Log.d("MicProbe", description);
        if (socketManager.mSocketSendMessages == null)
            return;
        long t = System.currentTimeMillis();
        RobotCommandOuterClass.RobotToServerMessage.Builder b =
                RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                        .setEventTime(Timestamp.newBuilder().setSeconds(t / 1000)
                                .setNanos((int) ((t % 1000) * 1000000)).build())
                        .setDescription(description);
        if (pcm != null) {
            b.setAudioProbe(ByteString.copyFrom(pcm));
            b.setAudioProbeChannels(channels);
            b.setAudioProbeRate(rate);
        }
        socketManager.sendAMessage(b.build(), socketManager.mSocketSendMessages);
    }
}
