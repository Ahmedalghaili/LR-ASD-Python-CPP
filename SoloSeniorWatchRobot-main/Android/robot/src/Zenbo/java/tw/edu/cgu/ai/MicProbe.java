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

    /** "audio_info": what the robot's Android says about its microphones. */
    void audioInfo(final android.content.Context context) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                StringBuilder b = new StringBuilder();
                try {
                    android.media.AudioManager am =
                            (android.media.AudioManager) context.getSystemService(android.content.Context.AUDIO_SERVICE);
                    for (android.media.AudioDeviceInfo d : am.getDevices(android.media.AudioManager.GET_DEVICES_INPUTS))
                        b.append("input id=").append(d.getId()).append(" type=").append(d.getType())
                                .append(" (").append(typeName(d.getType())).append(") name=").append(d.getProductName())
                                .append(" channel_counts=").append(java.util.Arrays.toString(d.getChannelCounts()))
                                .append(" index_masks=").append(java.util.Arrays.toString(d.getChannelIndexMasks()))
                                .append(" rates=").append(java.util.Arrays.toString(d.getSampleRates()))
                                .append("\n");
                } catch (Throwable e) {
                    b.append("getDevices failed: ").append(e).append("\n");
                }
                send("audio_info devices\n" + b, null, 0, 0);
                send("audio_info getprop\n" + shell("getprop", "audio|mic|dmic|snd|alsa|asus|ro.product|ro.hardware|ro.board|ro.build.fingerprint|voice|record"), null, 0, 0);
                send("audio_info /dev/snd\n" + list("/dev/snd"), null, 0, 0);
                String[] files = {"/proc/asound/version", "/proc/asound/cards", "/proc/asound/devices", "/proc/asound/pcm"};
                for (String f : files)
                    send("audio_info file=" + f + "\n" + read(f, 20000), null, 0, 0);
                // Capture streams of every sound card.
                java.io.File asound = new java.io.File("/proc/asound");
                java.io.File[] cards = asound.listFiles();
                if (cards != null)
                    for (java.io.File card : cards) {
                        if (!card.getName().startsWith("card"))
                            continue;
                        java.io.File[] pcms = card.listFiles();
                        if (pcms == null)
                            continue;
                        for (java.io.File pcm : pcms)
                            if (pcm.getName().startsWith("pcm") && pcm.getName().endsWith("c"))
                                send("audio_info file=" + pcm + "/info\n" + read(pcm + "/info", 4000)
                                        + "\n-- sub0/info --\n" + read(pcm + "/sub0/info", 4000), null, 0, 0);
                    }
                // Audio policy and mixer configuration (which microphones are wired how).
                int total = 0;
                for (String dir : new String[]{"/system/etc", "/vendor/etc", "/odm/etc"}) {
                    java.io.File[] etc = new java.io.File(dir).listFiles();
                    if (etc == null)
                        continue;
                    for (java.io.File f : etc) {
                        String n = f.getName().toLowerCase();
                        if (f.isFile() && (n.contains("audio") || n.contains("mixer") || n.contains("mic"))
                                && (n.endsWith(".conf") || n.endsWith(".xml") || n.endsWith(".cfg")) && total < 400000) {
                            String text = read(f.getPath(), 120000);
                            total += text.length();
                            send("audio_info file=" + f.getPath() + "\n" + text, null, 0, 0);
                        }
                    }
                }
                send("audio_info done", null, 0, 0);
            }
        }, "AudioInfo").start();
    }

    /** "mic_probe_devices": record 1 s from every input device directly (2 and 4 channels). */
    void probeDevices(final android.content.Context context) {
        runExclusive(new Runnable() {
            @Override
            public void run() {
                android.media.AudioManager am =
                        (android.media.AudioManager) context.getSystemService(android.content.Context.AUDIO_SERVICE);
                for (android.media.AudioDeviceInfo d : am.getDevices(android.media.AudioManager.GET_DEVICES_INPUTS))
                    for (int rate : RATES)
                        for (String setup : new String[]{"index2", "index4", "index8"})
                            for (int s : new int[]{0, 2})   // MIC, VOICE_RECOGNITION
                                record("mic_probe_dev", s, setup, rate, 1.0f, d,
                                        " device=" + d.getId() + " type=" + typeName(d.getType()));
                send("mic_probe_done", null, 0, 0);
            }
        });
    }

    private static String typeName(int type) {
        switch (type) {
            case android.media.AudioDeviceInfo.TYPE_BUILTIN_MIC: return "BUILTIN_MIC";
            case android.media.AudioDeviceInfo.TYPE_USB_DEVICE: return "USB_DEVICE";
            case android.media.AudioDeviceInfo.TYPE_USB_ACCESSORY: return "USB_ACCESSORY";
            case android.media.AudioDeviceInfo.TYPE_WIRED_HEADSET: return "WIRED_HEADSET";
            case android.media.AudioDeviceInfo.TYPE_BLUETOOTH_SCO: return "BLUETOOTH_SCO";
            case android.media.AudioDeviceInfo.TYPE_TELEPHONY: return "TELEPHONY";
            case android.media.AudioDeviceInfo.TYPE_FM_TUNER: return "FM_TUNER";
            case android.media.AudioDeviceInfo.TYPE_TV_TUNER: return "TV_TUNER";
            case android.media.AudioDeviceInfo.TYPE_LINE_ANALOG: return "LINE_ANALOG";
            case android.media.AudioDeviceInfo.TYPE_LINE_DIGITAL: return "LINE_DIGITAL";
            case android.media.AudioDeviceInfo.TYPE_IP: return "IP";
            default: return "TYPE_" + type;
        }
    }

    private static String read(String path, int max) {
        try {
            java.io.FileInputStream in = new java.io.FileInputStream(path);
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0 && out.size() < max)
                out.write(buf, 0, n);
            in.close();
            return out.toString("UTF-8");
        } catch (Throwable e) {
            return "(cannot read: " + e.getMessage() + ")";
        }
    }

    private static String list(String dir) {
        String[] names = new java.io.File(dir).list();
        return names == null ? "(cannot list)" : java.util.Arrays.toString(names);
    }

    private static String shell(String command, String filter) {
        try {
            Process p = Runtime.getRuntime().exec(command);
            java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream()));
            StringBuilder b = new StringBuilder();
            java.util.regex.Pattern keep = java.util.regex.Pattern.compile(filter, java.util.regex.Pattern.CASE_INSENSITIVE);
            String line;
            while ((line = r.readLine()) != null)
                if (keep.matcher(line).find())
                    b.append(line).append("\n");
            p.waitFor();
            return b.toString();
        } catch (Throwable e) {
            return "(" + command + " failed: " + e + ")";
        }
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
        record(tag, sourceIndex, setup, rate, seconds, null, "");
    }

    private void record(String tag, int sourceIndex, String setup, int rate, float seconds,
                        android.media.AudioDeviceInfo device, String label) {
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
            if (device != null && !recorder.setPreferredDevice(device))
                label += " preferred_device=refused";
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
        String routed = "";
        if (device != null && recorder != null) {
            try {
                android.media.AudioDeviceInfo used = recorder.getRoutedDevice();
                routed = " routed=" + (used == null ? "none" : used.getId());
            } catch (Throwable ignored) {
            }
        }
        send(tag + label + " source=" + SOURCE_NAMES[sourceIndex] + " setup=" + setup + " channels=" + channels
                + " rate=" + actualRate + routed + " result=" + result, pcm, channels, actualRate);
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
