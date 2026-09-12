"""Local-only demo server. No arbitrary file path or shell execution endpoints."""
import base64
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from PIL import Image, UnidentifiedImageError
from .data import list_images
from .model import Detector, MODES
from .reporting import png_data, views

MAX_BODY = 12 * 1024 * 1024

def create_server(model_path, samples, port=8765):
    model = Detector.load(model_path)
    paths = [p.resolve() for p in list_images(samples)]
    sample_root = Path(samples).resolve()
    uploads = Path(model_path).resolve().parent/"uploads"
    lock = Lock()
    web = Path(__file__).parent/"web"
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route=urlparse(self.path).path
            if route=="/":
                return self.reply(200,(web/"index.html").read_bytes(),"text/html; charset=utf-8")
            if route=="/api/info":
                metrics_path=Path(model_path).parent/"metrics.json"
                metrics=json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
                return self.reply(200,{"config":model.config,"splits":model.split_counts,"metrics":metrics,
                                      "samples":[{"id":i,"name":p.relative_to(sample_root).as_posix()} for i,p in enumerate(paths)]})
            return self.reply(404,{"error":"Not found"})

        def do_POST(self):
            if urlparse(self.path).path!="/api/predict":
                return self.reply(404,{"error":"Not found"})
            origin=self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self.reply(403,{"error":"Only same-origin local requests are allowed"})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if length<1 or length>MAX_BODY:
                    return self.reply(413,{"error":"Request must be 1 byte to 12 MiB"})
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):
                    raise ValueError("JSON object required")
                mode=payload.get("mode","gated")
                if mode not in MODES:
                    raise ValueError("Invalid detection mode")
                if "image_base64" in payload:
                    raw=base64.b64decode(payload["image_base64"],validate=True)
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.width*im.height>16_000_000:
                            raise ValueError("Image exceeds 16 megapixels")
                        im.load()
                        normalized=im.convert("RGB")
                    uploads.mkdir(parents=True,exist_ok=True)
                    path=uploads/(hashlib.sha256(raw).hexdigest()+".png")
                    normalized.save(path)
                else:
                    idx=payload.get("sample_id")
                    if type(idx) is not int or not 0<=idx<len(paths):
                        raise ValueError("Invalid sample id")
                    path=paths[idx]
                with lock:
                    result=model.predict(path,mode)
                    images={name:png_data(im) for name,im in views(path,result,model.config["size"]).items()}
                public={k:v for k,v in result.items() if k not in ("heatmap","global_map","spatial_map")}
                public["evidence"]["normal_image"]=Path(public["evidence"]["normal_image"]).name
                public["images"]=images
                return self.reply(200,public)
            except (ValueError, TypeError, KeyError, UnidentifiedImageError, OSError) as exc:
                return self.reply(400,{"error":str(exc)})
            except Exception:
                import traceback
                traceback.print_exc()
                return self.reply(500,{"error":"Prediction failed; check model and server console"})

        def log_message(self, fmt, *args):
            print("HTTP",fmt%args,flush=True)

    return ThreadingHTTPServer(("127.0.0.1",port),Handler)

def serve(model_path,samples,port=8765):
    server=create_server(model_path,samples,port)
    print(f"DefectScope: http://127.0.0.1:{server.server_port}",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
