"""
Simple HTTP upload server for Drishti CLIP inference.
Run: python server.py
"""

import http.server
import json
import os
from email import message_from_bytes
from email.policy import HTTP
import shutil
from pathlib import Path
from urllib.parse import parse_qs

from src.clip_inference import run_query, init

PORT = 8000

# ── Import your existing inference logic ──────────────────────────────────────
# Adjust this import to match your actual module/function names
try:
    from clip_inference import load_model, create_output, labels
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False
    print("Warning: clip_inference module not found. Running in save-only mode.")


class UploadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            # Serve the HTML file from the same directory as server.py
            #html_path = Path(__file__).parent / "upload.html"
            html_path = Path(__file__).parent / "pages/upload.html"
            with open(html_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        """Handle CORS preflight so the HTML page can call the API."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path != "/upload":
            self._json(404, {"error": "not found"})
            return

        # Parse multipart/form-data
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "expected multipart/form-data"})
            return


        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        raw = f"Content-Type: {content_type}\r\n\r\n".encode() + body
        msg = message_from_bytes(raw, policy=HTTP)

        parts = {}
        for part in msg.get_payload():
            disposition = part.get("Content-Disposition", "")
            # Extract name
            name_start = disposition.find('name="') + 6
            name_end   = disposition.find('"', name_start)
            name       = disposition[name_start:name_end]

            # Extract filename if present
            if 'filename="' in disposition:
                fn_start = disposition.find('filename="') + 10
                fn_end   = disposition.find('"', fn_start)
                parts[name] = (disposition[fn_start:fn_end], part.get_payload(decode=True))
            else:
                parts[name] = part.get_payload(decode=True).decode()


        model     = parts.get("model", "ViT-B/32")
        flavor     = parts.get("tuned", "base")
        labels    = json.loads(parts.get("labels", "[]"))
        print(f"model = {model} tuned = {flavor} labels = {labels}")

        # ── Get uploaded file ─────────────────────────────────────────────────
        if "image" not in parts and "server-image-name" not in parts:
            self._json(400, {"error": "no image field in request"})
            return
        query_path = ""
        if "image" in parts:
            filename, file_bytes = parts["image"]
            filename  = Path(filename).name
            directory = parts.get("directory", "data/inference/query_images")
        # ── Save to disk ──────────────────────────────────────────────────────
            save_dir = Path(directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            query_path = save_dir / filename
            with open(query_path, "wb") as f:
                f.write(file_bytes)
            print(f"Saved: {query_path}")
        elif "server-image-name" in parts:
            filename = parts["server-image-name"]
            directory = parts.get("directory", "data/inference/query_images")
            image_directory = Path(directory)
            query_path = image_directory / filename

        print(f"query image path = {query_path}")
        weights_path = parts.get("weights_path", "")


        # ── Run CLIP inference ────────────────────────────────────────────────
        result = {"filename": str(query_path), "matches": []}
        output = run_query(model, flavor, weights_path, labels, query_path)
        print(f"output = {output}")
        if output is not None:
            result = output
        self._json(200, result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


if __name__ == "__main__":
    init()
    server = http.server.HTTPServer(("0.0.0.0", PORT), UploadHandler)
    print(f"Server running at http://0.0.0.0:{PORT}")
    print(f"CLIP inference: {'enabled' if MODEL_LOADED else 'disabled (save-only mode)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
