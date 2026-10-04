#!/bin/bash
# Serve the published Zenbo app (publish_zenbo_v2.sh) on port 8000, all networks.
UPDATES="${ZENBO_UPDATES_DIR:-$HOME/SoloSeniorWatchRobot_build/app_updates}"
mkdir -p "$UPDATES" && cd "$UPDATES"
echo "serving $UPDATES on port 8000; robot app: http://$(hostname -I | tr ' ' '\n' | grep -m1 '^192\.'):8000/ZenboV2_latest.apk"
exec python3 -c "
import http.server, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.apk': 'application/vnd.android.package-archive'}
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
class S(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
S(('0.0.0.0', 8000), H).serve_forever()
"
