package tw.edu.cgu.ai;

import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.hardware.usb.UsbConstants;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.util.Log;

import com.google.protobuf.Timestamp;

import java.util.ArrayList;
import java.util.List;

/**
 * Reads the voice direction of a reSpeaker XVF3800 4-mic array plugged into
 * Zenbo's USB port (USB firmware) and sends it to the server, which turns
 * Zenbo toward the voice (ACTIVE_SPEAKER_DOA=respeaker).
 *
 * DOA_VALUE is read with a vendor control transfer (resid 20, cmd 18): a
 * status byte (0 = ok, 64 = retry), the angle (uint16) and a speech flag.
 * While the array hears speech each reading is sent as description
 * "respeaker_doa" with doa = the free-running beam's angle (AEC_AZIMUTH_VALUES,
 * resid 33, cmd 75); when speech ends one reading
 * with vad_status "quiet" follows. Changes of state (found, no permission,
 * lost, not plugged in) are sent as text "respeaker_usb ...".
 * The direction is also shown on the robot's panel, with the front and mirror
 * the server uses (poc_action "respeaker_front:<array angle>:<mirror 0/1>").
 */
class ReSpeakerUsb {
    private static final int VENDOR = 0x2886, PRODUCT = 0x001A;
    private static final String ACTION_PERMISSION = "tw.edu.cgu.ai.RESPEAKER_PERMISSION";

    private final Context context;
    private final SocketManager socketManager;
    private final ZenboCallback callback;
    private volatile boolean running = true;
    volatile VoiceDirectionView view;
    volatile int front = 90;            // array angle of Zenbo's front
    volatile boolean mirror = false;
    private String lastState = "";
    private long lastStateMs = 0;
    private int readings = 0, speechReadings = 0;

    ReSpeakerUsb(Context context, SocketManager socketManager, ZenboCallback callback) {
        this.context = context.getApplicationContext();
        this.socketManager = socketManager;
        this.callback = callback;
        new Thread(new Runnable() {
            @Override
            public void run() {
                // A crash here used to stop the reSpeaker silently: report it and start again.
                while (running) {
                    try {
                        loop();
                    } catch (Throwable e) {
                        lastState = "";
                        state("crashed: " + e, "error");
                        sleep(1000);
                    }
                }
            }
        }, "ReSpeakerUsb").start();
    }

    void stop() {
        running = false;
    }

    /** Sends state s to the server (when it changes, and every 10 s with the
     *  reading counts) and shows the short text on the panel title. */
    private void state(String s, String panel) {
        long now = System.currentTimeMillis();
        if (s.equals(lastState) && now - lastStateMs < 10000)
            return;
        if (s.equals(lastState)) {
            callback.sendText("respeaker_usb alive: " + s + "; readings " + readings + ", with speech " + speechReadings);
            lastStateMs = now;
            readings = speechReadings = 0;
            return;
        }
        lastState = s;
        lastStateMs = now;
        Log.d("ReSpeakerUsb", s);
        VoiceDirectionView v = view;
        if (v != null)
            v.setSource("reSpeaker: " + panel);
        callback.sendText("respeaker_usb " + s);
    }

    private void loop() {
        UsbManager usb = (UsbManager) context.getSystemService(Context.USB_SERVICE);
        boolean asked = false;
        long lastReportMs = 0;
        while (running) {
            UsbDevice device = null;
            List<String> seen = new ArrayList<>();
            for (UsbDevice d : usb.getDeviceList().values()) {
                seen.add(String.format("%04x:%04x", d.getVendorId(), d.getProductId()));
                if (d.getVendorId() == VENDOR && d.getProductId() == PRODUCT)
                    device = d;
            }
            if (device == null) {
                asked = false;
                // Repeat now and then, in case the server connected later.
                if (System.currentTimeMillis() - lastReportMs > 30000) {
                    lastState = "";
                    lastReportMs = System.currentTimeMillis();
                }
                state("not found; USB devices on Zenbo: " + seen, "not plugged in");
                sleep(1000);
                continue;
            }
            if (!usb.hasPermission(device)) {
                state("found, waiting for permission (tap OK on Zenbo's screen)", "tap OK to allow");
                if (!asked) {
                    usb.requestPermission(device, PendingIntent.getBroadcast(
                            context, 0, new Intent(ACTION_PERMISSION), 0));
                    asked = true;
                }
                sleep(1000);
                continue;
            }
            UsbDeviceConnection connection = usb.openDevice(device);
            if (connection == null) {
                state("found but cannot open it", "cannot open");
                sleep(1000);
                continue;
            }
            state("connected: reading the voice direction", "connected");
            read(connection);
            connection.close();
            state("lost", "lost");
        }
    }

    /** Reads `length` payload bytes of a parameter into buf (status byte
     *  first); false if the array does not answer. */
    private static boolean readParam(UsbDeviceConnection connection, int resid, int cmd, byte[] buf) {
        int n = 0;
        for (int attempt = 0; attempt < 20; ++attempt) {
            n = connection.controlTransfer(
                    UsbConstants.USB_DIR_IN | UsbConstants.USB_TYPE_VENDOR, 0,
                    0x80 | cmd, resid, buf, buf.length, 500);
            if (n != buf.length || buf[0] == 0)
                break;
            sleep(2);   // status 64: not ready yet
        }
        return n == buf.length && buf[0] == 0;
    }

    /** Float i of a parameter read with readParam, in degrees 0..360 (-1 = none). */
    private static String degrees(byte[] buf, int i) {
        int bits = (buf[1 + 4 * i] & 0xff) | (buf[2 + 4 * i] & 0xff) << 8
                | (buf[3 + 4 * i] & 0xff) << 16 | (buf[4 + 4 * i] & 0xff) << 24;
        float r = Float.intBitsToFloat(bits);
        if (Float.isNaN(r))
            return "-1";
        return String.valueOf(Math.round((Math.toDegrees(r) % 360 + 360) % 360));
    }

    /** Polls DOA_VALUE until the array stops answering. */
    private void read(UsbDeviceConnection connection) {
        byte[] buf = new byte[5];
        byte[] selected = new byte[9];  // AUDIO_MGR_SELECTED_AZIMUTHS: processed DoA, auto beam
        byte[] beams = new byte[17];    // AEC_AZIMUTH_VALUES: beam 1, beam 2, free-running, auto
        boolean speaking = false;
        int failures = 0;
        while (running && failures < 10) {
            if (!readParam(connection, 20, 18, buf)) {
                failures++;
                sleep(100);
                continue;
            }
            failures = 0;
            state("connected: reading the voice direction", "connected");
            readings++;
            int doaValue = (buf[1] & 0xff) | (buf[2] & 0xff) << 8;
            boolean speech = buf[3] != 0;
            if (speech)
                speechReadings++;
            // Inside Zenbo DOA_VALUE stays stuck on one angle (Zenbo's own
            // noise), while the free-running beam follows the voice
            // (tests/respeaker_compare_doa.py), so the direction sent is the
            // free-running beam's; DOA_VALUE only gives the speech flag.
            int angle = -1;
            if (speech && readParam(connection, 35, 11, selected) && readParam(connection, 33, 75, beams)) {
                String free = degrees(beams, 2);
                angle = Integer.parseInt(free);
                callback.sendText("respeaker_usb raw doa=" + doaValue + " processed=" + degrees(selected, 0)
                        + " auto=" + degrees(selected, 1) + " free=" + free);
            }
            if (speech && angle < 0) {
                speaking = speech;
                sleep(100);
                continue;
            }
            if (speech || speaking)
                send(Math.max(angle, 0), speech);
            VoiceDirectionView v = view;
            if (speech && v != null) {
                int d = ((angle - front) % 360 + 540) % 360 - 180;   // + = Zenbo's left
                v.onDoa(mirror ? -d : d);
                v.onVoiceEvent("speech (reSpeaker)");
            }
            speaking = speech;
            sleep(100);
        }
    }

    private void send(int angle, boolean speech) {
        if (socketManager.mSocketSendMessages == null)
            return;
        long t = System.currentTimeMillis();
        socketManager.sendAMessage(RobotCommandOuterClass.RobotToServerMessage.newBuilder()
                .setEventTime(Timestamp.newBuilder().setSeconds(t / 1000)
                        .setNanos((int) ((t % 1000) * 1000000)).build())
                .setDescription("respeaker_doa")
                .setDoa(angle)
                .setVadStatus(speech ? "speech" : "quiet")
                .build(), socketManager.mSocketSendMessages);
    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ignored) {
        }
    }
}
