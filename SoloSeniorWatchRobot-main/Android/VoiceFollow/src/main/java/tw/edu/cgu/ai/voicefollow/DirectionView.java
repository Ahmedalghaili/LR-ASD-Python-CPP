package tw.edu.cgu.ai.voicefollow;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Full-screen view: Zenbo seen from above (front up) with four coloured sides.
 * The side of the last voice direction lights up in its colour (bright for
 * 4 s, then lighter until the next one). Right half: the direction, the
 * listening status, the last directions and Zenbo's events.
 * Sides assume + = Zenbo's left (measured: left +90, right -90, behind 170).
 */
class DirectionView extends View {
    private static final String[] NAMES = {"FRONT", "LEFT", "BEHIND", "RIGHT"};
    private static final int[] COLORS = {
            Color.rgb(40, 120, 230), Color.rgb(40, 170, 80), Color.rgb(140, 70, 200), Color.rgb(240, 130, 20)};

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<String> events = new ArrayList<>();
    private final List<String> directions = new ArrayList<>();
    private final SimpleDateFormat clock = new SimpleDateFormat("HH:mm:ss", Locale.US);
    private int doa = 0, doaCount = 0, eventCount = 0;
    private long doaTime = 0;
    private boolean listening = false;
    private String status = "Starting...";

    DirectionView(Context context) {
        super(context);
        setClickable(true);
    }

    static int sector(int degrees) {
        int d = ((degrees % 360) + 540) % 360 - 180;   // -180..179
        if (Math.abs(d) <= 45) return 0;
        if (d > 45 && d < 135) return 1;
        if (d < -45 && d > -135) return 3;
        return 2;
    }

    synchronized void onDoa(int degrees) {
        doa = degrees;
        doaCount++;
        doaTime = System.currentTimeMillis();
        directions.add(0, clock.format(new Date()) + "  " + NAMES[sector(degrees)] + "  (" + degrees + " deg)");
        if (directions.size() > 6)
            directions.remove(directions.size() - 1);
        addEventLocked("onDOA " + degrees);
    }

    synchronized void addEvent(String text) {
        addEventLocked(text);
    }

    private void addEventLocked(String text) {
        eventCount++;
        events.add(0, clock.format(new Date()) + "  " + text);
        if (events.size() > 8)
            events.remove(events.size() - 1);
        postInvalidate();
    }

    synchronized void setStatus(boolean on, String text) {
        listening = on;
        status = text;
        postInvalidate();
    }

    @Override
    protected synchronized void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        c.drawColor(Color.WHITE);
        paint.setStyle(Paint.Style.FILL);

        // Left half: the robot from above with four sides.
        float cx = w * 0.27f, cy = h * 0.52f, r = Math.min(w * 0.24f, h * 0.42f);
        boolean fresh = doaCount > 0 && System.currentTimeMillis() - doaTime < 4000;
        int lit = doaCount > 0 ? sector(doa) : -1;
        float[] start = {-135, 135, 45, -45};   // canvas angles: 0 = right, 90 = down
        RectF box = new RectF(cx - r, cy - r, cx + r, cy + r);
        for (int s = 0; s < 4; s++) {
            int col = COLORS[s];
            paint.setStyle(Paint.Style.FILL);
            if (s == lit)
                paint.setColor(fresh ? col : Color.argb(110, Color.red(col), Color.green(col), Color.blue(col)));
            else
                paint.setColor(Color.rgb(232, 232, 232));
            c.drawArc(box, start[s], 90, true, paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(6);
            paint.setColor(Color.WHITE);
            c.drawArc(box, start[s], 90, true, paint);
        }
        paint.setStyle(Paint.Style.FILL);
        paint.setTextAlign(Paint.Align.CENTER);
        paint.setTextSize(r * 0.13f);
        for (int s = 0; s < 4; s++) {
            double a = Math.toRadians(new double[]{0, 90, 180, -90}[s]);   // + = left
            float tx = (float) (cx - r * 0.66 * Math.sin(a)), ty = (float) (cy - r * 0.66 * Math.cos(a));
            paint.setColor(s == lit ? Color.WHITE : COLORS[s]);
            c.drawText(NAMES[s], tx, ty + r * 0.05f, paint);
        }
        paint.setColor(Color.rgb(40, 40, 40));
        c.drawCircle(cx, cy, r * 0.2f, paint);
        paint.setColor(Color.WHITE);
        paint.setTextSize(r * 0.09f);
        c.drawText("ZENBO", cx, cy + r * 0.03f, paint);
        if (doaCount > 0) {
            double a = Math.toRadians(doa);
            paint.setStrokeWidth(8);
            paint.setColor(Color.rgb(30, 30, 30));
            c.drawLine(cx, cy, (float) (cx - r * 0.98 * Math.sin(a)), (float) (cy - r * 0.98 * Math.cos(a)), paint);
        }
        paint.setTextAlign(Paint.Align.LEFT);

        // Right half: text.
        float x = w * 0.56f, y = h * 0.1f;
        paint.setColor(Color.rgb(30, 30, 30));
        paint.setTextSize(h * 0.05f);
        c.drawText("Zenbo Voice Follow", x, y, paint);
        y += h * 0.09f;
        paint.setTextSize(h * 0.065f);
        paint.setColor(doaCount == 0 ? Color.rgb(120, 120, 120) : COLORS[sector(doa)]);
        c.drawText(doaCount == 0 ? "No direction yet" : NAMES[sector(doa)] + "   " + doa + " deg", x, y, paint);
        y += h * 0.06f;
        paint.setTextSize(h * 0.035f);
        paint.setColor(listening ? Color.rgb(20, 120, 50) : Color.rgb(180, 60, 40));
        c.drawText(status, x, y, paint);
        y += h * 0.05f;
        paint.setColor(Color.rgb(60, 60, 60));
        c.drawText("Directions: " + doaCount + "    Zenbo events: " + eventCount, x, y, paint);
        y += h * 0.07f;
        paint.setTextSize(h * 0.032f);
        c.drawText("Last directions:", x, y, paint);
        for (String d : directions) {
            y += h * 0.042f;
            String side = d.split("  ")[1];
            int idx = java.util.Arrays.asList(NAMES).indexOf(side);
            paint.setColor(idx >= 0 ? COLORS[idx] : Color.DKGRAY);
            c.drawText(d, x + 20, y, paint);
        }
        y += h * 0.07f;
        paint.setColor(Color.rgb(60, 60, 60));
        c.drawText("Zenbo events:", x, y, paint);
        paint.setTextSize(h * 0.028f);
        for (String e : events) {
            y += h * 0.036f;
            c.drawText(e, x + 20, y, paint);
        }
        if (fresh)
            postInvalidateDelayed(500);
    }
}
