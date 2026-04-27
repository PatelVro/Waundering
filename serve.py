"""HTTP server with Cache-Control: no-cache on every response."""
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4173


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise


with http.server.HTTPServer(("", PORT), NoCacheHandler) as httpd:
    print(f"Serving on :{PORT} (no-cache)", flush=True)
    httpd.serve_forever()
