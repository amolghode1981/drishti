"""
Simple HTTP upload server for Drishti CLIP inference.
Run: python server.py
"""

import http.server
import json
import os
from email import message_from_bytes
from email.policy import HTTP
from pydoc import doc
import shutil
from pathlib import Path
from urllib.parse import parse_qs

from numpy import object_


from src.clip_inference import run_query, init as clip_init
from src.web.db import db_connection_check, list_images_from_two_step_collection, write_to_persistent_storage, get_image, list_images, get_image_from_two_step_collection
from src.web import analysis
from src.web.serverutils import read_image_object_from_storage, write_image_to_storage, read_image_from_storage
from src.clip_grdino_inference import run_composite_query, delete_intermediate_files
import threading
PORT = 8000

# def write_to_persistent_storage(filename, model, flavor,query_path, response, labels):
#     try:
#         checksum = sha256_of_file(query_path)
#         upsert_reading(
#             checksum = checksum,
#             filename = filename,
#             model    = model,
#             flavor   = flavor,
#             labels   = labels,
#             matches  = response.get("matches", []),
#         )
#         print(f"Saved reading to MongoDB: {checksum[:12]}...")
#         return checksum
#     except Exception as e:
#         print(f"MongoDB write failed: {e}")
#         return "Error: MongoWrite failed"

class UploadHandler(http.server.BaseHTTPRequestHandler):
    def _serve_page(self, relative_path: str):
        html_path = Path(__file__).parent / relative_path
        if not html_path.exists():
            self._json(404, {"error": f"{relative_path} not found"})
            return
        with open(html_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def __run_query(self, model, flavor, weights_path, labels, query_path):
        try:
            output = run_query(model, flavor, weights_path, labels, query_path)
            return output
        except Exception as e:
            raise RuntimeError(f"CLIP inference failed: {e}")

    def __run_composite_query(self, model, flavor, weights_path, labels, query_path):
        try:
            output = run_composite_query(model, flavor, weights_path, labels, query_path)
            return output
        except Exception as e:
            raise RuntimeError(f"CLIP inference failed: {e}")

    def __process_batch(self, directory: str, model:str, count: int):
        labels = [
            # Musical
            {'label': 'A person holding Indian Veena without any shields or swords nearby and performing music', 'category': 'Musical'},
            {'label': 'A woman or a man with Indian Lute, Veena',                                               'category': 'Musical'},
            {'label': 'People playing music in a courtyard',                                                    'category': 'Musical'},
            {'label': 'A person playing musical drums',                                                         'category': 'Musical'},
            {'label': 'A person playing an Indian stringed instrument like Veena or Sitar',                     'category': 'Musical'},
            {'label': 'A person holding musical instruments while performing music',                            'category': 'Musical'},
            {'label': 'A court scene with one or many musical instruments',                                     'category': 'Musical'},
            {'label': 'A court scene with musicians and dancers entertaining people',                           'category': 'Musical'},
            {'label': 'A man practicing music alone holding a long necked gaurd lute',                          'category': 'Musical'},

            # Non-Musical
            {'label': 'A court scene without any musical instruments',                                          'category': 'Non-Musical'},
            {'label': 'Soldiers in combat with weapons',                                                        'category': 'Non-Musical'},
            {'label': 'A farmer working in a field',                                                            'category': 'Non-Musical'},
            {'label': 'A forest or natural landscape without any musical instrument',                           'category': 'Non-Musical'},
            {'label': 'A gathering with no musical instruments',                                                'category': 'Non-Musical'},
            {'label': 'A people holding weapons',                                                               'category': 'Non-Musical'},
            {'label': 'A person holding plough',                                                                'category': 'Non-Musical'},
            {'label': 'A person carrying a curved agricultural tool or plow on their back',                     'category': 'Non-Musical'},
            {'label': 'Figures in a scene with farming tools',                                                  'category': 'Non-Musical'},
        ]
        def __run_batch():
            for fname in os.listdir(directory):
                if not fname.lower().endswith((".jpg", ".jpeg")):
                    continue
                filepath = Path(directory) / fname
                # Run inference multiple times to find out p-value later.
                for i in range(count):
                    try:
                        output = self.__run_query(model, "base", "", labels, str(filepath))
                        if output is not None:
                            result = output
                        checksum = write_to_persistent_storage("images", fname, model, "base", str(filepath), result, labels)
                        print(f"checksum = {checksum} filename = {fname}")
                    except Exception as e:
                        print(f"CLIP inference failed for {fname}: {e}")
            print("Batch processing complete.")

        threading.Thread(target=__run_batch, daemon=True).start()
        self._json(200, {"status": "started"})

    def __process_composite_batch(self, directory: str, model:str, count: int):
        labels = [
            # Musical
            {'label': 'A person holding Indian Veena without any shields or swords nearby and performing music', 'category': 'Musical'},
            {'label': 'A woman or a man with Indian Lute, Veena',                                               'category': 'Musical'},
            {'label': 'People playing music in a courtyard',                                                    'category': 'Musical'},
            {'label': 'A person playing musical drums',                                                         'category': 'Musical'},
            {'label': 'A person playing an Indian stringed instrument like Veena or Sitar',                     'category': 'Musical'},
            {'label': 'A person holding musical instruments while performing music',                            'category': 'Musical'},
            {'label': 'A court scene with one or many musical instruments',                                     'category': 'Musical'},
            {'label': 'A court scene with musicians and dancers entertaining people',                           'category': 'Musical'},
            {'label': 'A man practicing music alone holding a long necked gaurd lute',                          'category': 'Musical'},

            # Non-Musical
            {'label': 'A court scene without any musical instruments',                                          'category': 'Non-Musical'},
            {'label': 'Soldiers in combat with weapons',                                                        'category': 'Non-Musical'},
            {'label': 'A woman holding a small pot, vessel, or container',     'category': 'Non-Musical'},
            {'label': 'A forest or natural landscape without any musical instrument',                           'category': 'Non-Musical'},
            {'label': 'A gathering with no musical instruments',                                                'category': 'Non-Musical'},
            {'label': 'A people holding weapons',                                                               'category': 'Non-Musical'},
            {'label': 'A person applying cosmetics or handling a small jar',   'category': 'Non-Musical'},
            {'label': 'A person carrying a curved agricultural tool or plow on their back',                     'category': 'Non-Musical'},
            {'label': 'A lady at her toilette with a mirror or small object',  'category': 'Non-Musical'},
        ]
        # labels = [
        #     # Musical
        #     {'label': 'A person holding musical instrument', 'category': 'Musical'},
        #     {'label': 'A person playing musical instrument', 'category': 'Musical'},
        #     {'label': 'A person without any weapons, swords, shields', 'category': 'Musical'},
        #     # Non-Musical
        #     {'label': 'A person without any musical instruments', 'category': 'Non-Musical'},
        #     {'label': 'A person with weapons', 'category': 'Non-Musical'},
        #     {'label': 'A forest or natural landscape without any musical instrument', 'category': 'Non-Musical'},
        # ]
        def __run_batch():
            loop_counter = 0
            for fname in os.listdir(directory):
                if not fname.lower().endswith((".jpg", ".jpeg")):
                    continue
                # if loop_counter >= 6:
                #     break
                # loop_counter += 1
                filepath = Path(directory) / fname
                # Run inference multiple times to find out p-value later.
                for i in range(count):
                    try:
                        output = self.__run_composite_query(model, "base", "", labels, str(filepath))
                        if output is not None:
                            result = output
                        for r in result:
                            object_path = r.get("image_object_path", "")
                            write_to_persistent_storage("two_step_images", fname, model, "base", object_path, r, labels)
                            #delete_intermediate_files()
                    except Exception as e:
                        print(f"CLIP inference failed for {fname}: {e}")
            print("Batch processing complete.")

        threading.Thread(target=__run_batch, daemon=True).start()
        self._json(200, {"status": "started"})

    def __process_upload(self):
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
            query_path = write_image_to_storage(filename, file_bytes)
        elif "server-image-name" in parts:
            filename = parts["server-image-name"]
            directory = parts.get("directory", "data/inference/query_images")
            image_directory = Path(directory)
            query_path = image_directory / filename

        print(f"query image path = {query_path}")
        weights_path = parts.get("weights_path", "")
        try:
            output = self.__run_query(model, flavor, weights_path, labels, query_path)
            print(f"output = {output}")
            if output is not None:
                result = output
            checksum = write_to_persistent_storage("images", filename, model, flavor, query_path, result, labels)
            result["checksum"] = checksum
            self._json(200, result)
            print(f"CLIP inference output: {output}")
        except Exception as e:
            self._json(500, {"error": f"CLIP inference failed: {e}"})
            return
        # # ── Run CLIP inference ────────────────────────────────────────────────
        # result = {"filename": str(query_path), "matches": []}
        # output = run_query(model, flavor, weights_path, labels, query_path)
        # print(f"output = {output}")
        # if output is not None:
        #     result = output
        # checksum = write_to_persistent_storage(filename, model, flavor, query_path, result, labels)
        # result["checksum"] = checksum
        # self._json(200, result)


    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_page("pages/upload.html")
        elif self.path == "/view" or self.path == "/view.html":
            self._serve_page("pages/view_records.html")
        elif self.path == "/view_composite" or self.path == "/view_composite.html":
            self._serve_page("pages/view_records_composite.html")
        elif self.path == "/batch" or self.path == "/batch.html":
            self._serve_page("pages/batch.html")
        elif self.path == "/composite_batch" or self.path == "/composite_batch.html":
            self._serve_page("pages/composite_batch.html")
        # ── API: single image by checksum ─────────────────────────────────────
        elif self.path.startswith("/api/image/"):
            checksum = self.path.split("/api/image/")[-1].strip("/")
            doc = get_image(checksum)
            b64_encoded_image = read_image_from_storage(doc["filename"])
            result = {
                "_id": str(doc["_id"]),
                "checksum": checksum,
                "filename": doc["filename"],
                "image_b64": b64_encoded_image,
            }
            image_analysis = analysis.Analysis(checksum)
            result["output"] = image_analysis.analyzeImage()
            #result["output"] = image_analysis.analyzeImageCombinedSignal()
            if doc is None:
                self._json(404, {"error": "image not found"})
            else:
                self._json(200, result)
        elif self.path.startswith("/api/composite_image/"):
            checksum = self.path.split("/api/composite_image/")[-1].strip("/")
            doc = get_image_from_two_step_collection(checksum)
            b64_encoded_image = read_image_object_from_storage(doc["image_object_path"])
            result = {
                "_id": str(doc["_id"]),
                "checksum": checksum,
                "filename": doc["filename"],
                "object_path": doc["image_object_path"],
                "image_b64": b64_encoded_image,
            }
            image_analysis = analysis.Analysis(checksum, composite=True)
            result["output"] = image_analysis.analyzeImage()
            #result["output"] = image_analysis.analyzeImageCombinedSignal()
            if doc is None:
                self._json(404, {"error": "image not found"})
            else:
                self._json(200, result)

        # ── API: list all images ──────────────────────────────────────────────
        elif self.path == "/api/images":
            docs = list_images()
            for d in docs:
                d["_id"] = str(d["_id"])
            self._json(200, {"images": docs})
        elif self.path == "/api/composite_images":
            docs = list_images_from_two_step_collection()
            for d in docs:
                d["_id"] = str(d["_id"])
            self._json(200, {"images": docs})

        # ── API: health check ─────────────────────────────────────────────────
        elif self.path == "/api/ping":
            self._json(200, {"ok": db_connection_check()})
        else:
            self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        """Handle CORS preflight so the HTML page can call the API."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/batch":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            directory = body.get("directory", "")
            model = body.get("model", "ViT-B/32")
            count = int(body.get("count", 1))
            self.__process_batch(directory, model, count)
            return
        elif self.path == "/api/composite/batch":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            directory = body.get("directory", "")
            model = body.get("model", "ViT-B/32")
            count = int(body.get("count", 1))
            self.__process_composite_batch(directory, model, count)
            return
        elif self.path == "/upload":
            self.__process_upload()
            return
        else:
            self._json(404, {"error": "not found"})
            return

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

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

def init():
    clip_init()
    connected = db_connection_check()
    print(f"MongoDB: {'connected' if connected else 'FAILED — check MONGO_URI in src/db.py'}")

import socketserver

class FastHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]  # use IP directly, skip socket.getfqdn()
        self.server_port = self.server_address[1]

if __name__ == "__main__":
    init()

    server = FastHTTPServer(("0.0.0.0", PORT), UploadHandler)
    print(f"Server running at http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
