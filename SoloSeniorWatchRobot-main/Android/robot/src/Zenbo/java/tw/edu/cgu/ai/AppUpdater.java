package tw.edu.cgu.ai;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Updates the app from the PC, so a new version does not have to be copied
 * to the robot by hand.
 *
 * The PC serves http://SERVER:8000/latest.json ({"versionCode": N, "file":
 * "ZenboV2_latest.apk", "versionName": "..."}; see Android/publish_zenbo_v2.sh).
 * When N is newer than this app, the APK is downloaded and Android's install
 * screen is opened: the user only taps "Install". (Android 6 does not let an
 * ordinary app install silently.)
 */
class AppUpdater {
    static final int PORT = 8000;
    private static volatile boolean running = false;

    static void check(final Activity activity, final String server, final ZenboCallback report) {
        if (server == null || server.isEmpty() || running)
            return;
        running = true;
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    String base = "http://" + server + ":" + PORT + "/";
                    JSONObject latest = new JSONObject(new String(get(base + "latest.json"), "UTF-8"));
                    int code = latest.getInt("versionCode");
                    int mine = activity.getPackageManager().getPackageInfo(activity.getPackageName(), 0).versionCode;
                    if (code <= mine) {
                        say(report, "update_check up_to_date version=" + mine);
                        return;
                    }
                    final String name = latest.optString("versionName", String.valueOf(code));
                    say(report, "update_check downloading version=" + code + " (installed " + mine + ")");
                    byte[] apk = get(base + latest.getString("file"));
                    File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                    dir.mkdirs();
                    final File file = new File(dir, "ZenboNurseHelperV2_update.apk");
                    FileOutputStream out = new FileOutputStream(file);
                    out.write(apk);
                    out.close();
                    say(report, "update_check install_screen version=" + code);
                    activity.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(activity, "New version " + name + ": tap Install", Toast.LENGTH_LONG).show();
                            Intent intent = new Intent(Intent.ACTION_VIEW);
                            intent.setDataAndType(Uri.fromFile(file), "application/vnd.android.package-archive");
                            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                            activity.startActivity(intent);
                        }
                    });
                } catch (Exception e) {
                    // No update server on the PC is normal: the app works as before.
                    Log.d("AppUpdater", "no update: " + e);
                    say(report, "update_check failed " + e.getClass().getSimpleName() + ": " + e.getMessage());
                } finally {
                    running = false;
                }
            }
        }, "AppUpdater").start();
    }

    private static void say(ZenboCallback report, String text) {
        Log.d("AppUpdater", text);
        if (report != null)
            report.sendText(text);
    }

    private static byte[] get(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(3000);
        c.setReadTimeout(20000);
        c.setUseCaches(false);
        try {
            if (c.getResponseCode() != 200)
                throw new Exception("HTTP " + c.getResponseCode() + " for " + url);
            InputStream in = c.getInputStream();
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] chunk = new byte[65536];
            int n;
            while ((n = in.read(chunk)) > 0)
                buf.write(chunk, 0, n);
            return buf.toByteArray();
        } finally {
            c.disconnect();
        }
    }
}
