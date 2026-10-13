package tw.edu.cgu.ai;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/**
 * Voice direction test panel on the robot screen.
 *
 * Top: Zenbo's own direction of a voice (onDOA from its 4-microphone speech
 * system) as four sides around the robot. Zenbo's front is up; the side the
 * voice came from lights up for 4 s, with the raw angle. The sides assume
 * + = Zenbo's left; if a voice on the left lights "RIGHT", the sign is the
 * other way round.
 * Bottom: the two microphone channels this app records, with their levels
 * and whether they are the same signal (a copy) or different microphones.
 * Tap the panel to start or stop listening (Zenbo speakAndListen).
 */
class VoiceDirectionView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private volatile int doa = 0;
    private volatile long doaTime = 0;
    private volatile int doaCount = 0;
    private volatile double level0 = -120, level1 = -120;
    private volatile String channels = "-";
    private volatile String status = "Tap here to start listening";
    private volatile boolean listening = false;

    VoiceDirectionView(Context context) {
        super(context);
        setClickable(true);
    }

    /** A new direction from Zenbo (event_vad_status, vad_status "onDOA"). */
    void onDoa(int degrees) {
        doa = degrees;
        doaTime = System.currentTimeMillis();
        doaCount++;
        postInvalidate();
    }

    /** Levels (dBFS) of the two recorded channels and how they compare. */
    void setChannels(double db0, double db1, String comparison) {
        level0 = db0;
        level1 = db1;
        channels = comparison;
        postInvalidate();
    }

    void setListening(boolean on, String text) {
        listening = on;
        status = text;
        postInvalidate();
    }

    // Sector of a direction: 0 front, 1 left, 2 behind, 3 right (+ = left).
    private static int sector(int degrees) {
        int d = ((degrees % 360) + 540) % 360 - 180;   // -180..179
        if (Math.abs(d) <= 45) return 0;
        if (d > 45 && d < 135) return 1;
        if (d < -45 && d > -135) return 3;
        return 2;
    }

    @Override
    protected void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.argb(235, 250, 250, 250));
        c.drawRoundRect(new RectF(0, 0, w, h), 24, 24, paint);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3);
        paint.setColor(Color.rgb(120, 120, 120));
        c.drawRoundRect(new RectF(2, 2, w - 2, h - 2), 24, 24, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.rgb(30, 30, 30));
        paint.setTextSize(28);
        c.drawText("Voice direction (Zenbo 4 mics)", 20, 40, paint);

        // Four sides around the robot.
        float cx = w / 2f, cy = 70 + (h * 0.55f) / 2f, r = Math.min(w, h * 0.55f) / 2f - 10;
        boolean fresh = System.currentTimeMillis() - doaTime < 4000 && doaCount > 0;
        int lit = fresh ? sector(doa) : -1;
        String[] names = {"FRONT", "LEFT", "BEHIND", "RIGHT"};
        float[] start = {-135, 135, 45, -45};   // canvas angles: 0 = right, 90 = down
        RectF box = new RectF(cx - r, cy - r, cx + r, cy + r);
        for (int s = 0; s < 4; s++) {
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(s == lit ? Color.rgb(60, 190, 90) : Color.rgb(225, 225, 225));
            c.drawArc(box, start[s], 90, true, paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.WHITE);
            paint.setStrokeWidth(4);
            c.drawArc(box, start[s], 90, true, paint);
        }
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.rgb(40, 40, 40));
        c.drawCircle(cx, cy, r * 0.22f, paint);              // the robot
        paint.setColor(Color.WHITE);
        paint.setTextSize(22);
        paint.setTextAlign(Paint.Align.CENTER);
        c.drawText("ZENBO", cx, cy + 8, paint);
        paint.setColor(Color.rgb(30, 30, 30));
        paint.setTextSize(24);
        c.drawText(names[0], cx, cy - r * 0.6f, paint);
        c.drawText(names[2], cx, cy + r * 0.7f, paint);
        c.drawText(names[1], cx - r * 0.62f, cy + 8, paint);
        c.drawText(names[3], cx + r * 0.62f, cy + 8, paint);
        if (doaCount > 0) {
            // Arrow toward the raw angle (+ = left).
            double a = Math.toRadians(doa);
            float ex = (float) (cx - r * 0.95 * Math.sin(a)), ey = (float) (cy - r * 0.95 * Math.cos(a));
            paint.setStrokeWidth(6);
            paint.setColor(fresh ? Color.rgb(20, 110, 40) : Color.rgb(150, 150, 150));
            c.drawLine(cx, cy, ex, ey, paint);
        }
        paint.setTextAlign(Paint.Align.LEFT);

        // Text lines.
        float y = cy + r + 40;
        paint.setTextSize(26);
        paint.setColor(fresh ? Color.rgb(20, 110, 40) : Color.rgb(60, 60, 60));
        c.drawText(doaCount == 0 ? "No direction yet" :
                "Direction: " + doa + " deg  (" + names[sector(doa)] + ")  #" + doaCount, 20, y, paint);
        y += 38;
        paint.setColor(Color.rgb(60, 60, 60));
        paint.setTextSize(22);
        c.drawText("App mic channels:", 20, y, paint);
        drawBar(c, 230, y - 18, w - 250, 16, level0);
        drawBar(c, 230, y + 4, w - 250, 16, level1);
        y += 40;
        paint.setColor("DIFFERENT".equals(channels) ? Color.rgb(20, 110, 40) : Color.rgb(170, 60, 40));
        c.drawText("ch0 " + Math.round(level0) + " dB, ch1 " + Math.round(level1) + " dB: " + channels, 20, y, paint);
        y += 36;
        paint.setColor(listening ? Color.rgb(20, 110, 40) : Color.rgb(170, 60, 40));
        paint.setTextSize(24);
        c.drawText(status, 20, y, paint);
        // Keep the "fresh" highlight fading out.
        if (fresh)
            postInvalidateDelayed(500);
    }

    private void drawBar(Canvas c, float x, float y, float width, float height, double db) {
        float frac = (float) Math.max(0, Math.min(1, (db + 70) / 70));   // -70..0 dB
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.rgb(220, 220, 220));
        c.drawRect(x, y, x + width, y + height, paint);
        paint.setColor(Color.rgb(70, 140, 220));
        c.drawRect(x, y, x + width * frac, y + height, paint);
    }
}
