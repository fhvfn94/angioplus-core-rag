from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess

HOST = "aqua-floyd-appreciated-apparently.trycloudflare.com"
IP = "104.16.230.132"

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/tts":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        cmd = [
            "curl.exe",
            "-fsS",
            "--max-time", "120",
            "--resolve", f"{HOST}:443:{IP}",
            "-X", "POST",
            f"https://{HOST}/tts",
            "-H", "Content-Type: application/json",
            "--data-binary", "@-",
        ]

        p = subprocess.run(cmd, input=body, capture_output=True)

        if p.returncode != 0:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(p.stderr)
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(p.stdout)))
        self.end_headers()
        self.wfile.write(p.stdout)

        print(f"TTS OK: {len(p.stdout)} bytes")

    def log_message(self, format, *args):
        return

print("Local TTS proxy listening on http://127.0.0.1:8002")
HTTPServer(("127.0.0.1", 8002), Handler).serve_forever()
