"""Serve the local-only EPL match interface on loopback; no new dependencies."""

import argparse
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from interface.evidence import explain_match
from src.build_history import read_canonical_matches
from src.freeze_model import PROJECT_ROOT, verify_freeze
from src.predict import canonical_teams

STATIC = Path(__file__).resolve().parent / "static"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/styles.css": ("styles.css", "text/css; charset=utf-8")}


def metadata(root=PROJECT_ROOT):
    verify_freeze(root)
    canonical = read_canonical_matches(root / "data/processed/canonical_matches.csv")
    latest = date.fromisoformat(str(canonical.date.max()))
    return {"teams": sorted(canonical_teams(root)), "snapshot_date": latest.isoformat(),
            "minimum_date": (latest + timedelta(days=1)).isoformat(),
            "default_date": max(date.today(), latest + timedelta(days=1)).isoformat(),
            "scope": "Historical EPL clubs; a listed club is not confirmation of current EPL membership or a scheduled fixture."}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, *, root=PROJECT_ROOT, predict=explain_match):
        self.root = root
        self.predict = predict
        self.busy = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "EPLLocal/1.0"

    def log_message(self, *args):
        pass

    def respond(self, status, content, content_type="application/json; charset=utf-8"):
        # Drain a small rejected body before closing. Otherwise a client still
        # sending it can receive a TCP reset instead of the useful HTTP error.
        if self.command == "POST" and not getattr(self, "body_consumed", False):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if 0 < length <= 4096:
                    self.connection.settimeout(1)
                    self.rfile.read(length)
            except (OSError, ValueError):
                pass
            self.body_consumed = True
        body = json.dumps(content, allow_nan=False).encode() if isinstance(content, dict) else content
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_request(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed or (origin is not None and origin != f"http://{host}"):
            self.respond(403, {"error": "Only same-origin local requests are allowed."})
            return False
        return True

    def do_GET(self):
        if not self.allowed_request():
            return
        path = urlsplit(self.path).path
        if path in ASSETS:
            filename, content_type = ASSETS[path]
            self.respond(200, (STATIC / filename).read_bytes(), content_type)
        elif path == "/api/metadata":
            try:
                self.respond(200, metadata(self.server.root))
            except (RuntimeError, OSError, ValueError, KeyError):
                self.respond(503, {"error": "The frozen snapshot cannot be verified. Check the model bundle and run the freeze verifier in the project."})
        else:
            self.respond(404, {"error": "Not found."})

    def do_POST(self):
        if not self.allowed_request():
            return
        if self.path != "/api/predict":
            self.respond(404, {"error": "Not found."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond(415, {"error": "Use JSON for prediction requests."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError("Request body must be between 1 and 2048 bytes.")
            self.connection.settimeout(10)
            body = self.rfile.read(length)
            self.body_consumed = True
            request = json.loads(body)
            if not isinstance(request, dict) or set(request) != {"home", "away", "date"}:
                raise ValueError("Choose a home team, an away team, and a date.")
            if any(not isinstance(value, str) or not 1 <= len(value) <= 100 for value in request.values()):
                raise ValueError("Team names and date must be short text values.")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
            return
        if not self.server.busy.acquire(blocking=False):
            self.respond(429, {"error": "A prediction is already running. Try again shortly."})
            return
        try:
            result = self.server.predict(request["home"], request["away"], request["date"], root=self.server.root)
            self.respond(200, result)
        except (RuntimeError, ValueError, KeyError) as exc:
            self.respond(400, {"error": str(exc)})
        except OSError:
            self.respond(503, {"error": "The frozen model or data could not be read. Check the local project setup."})
        finally:
            self.server.busy.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    try:
        with LocalServer(("127.0.0.1", args.port)) as server:
            print(f"Local EPL interface: http://127.0.0.1:{args.port}", flush=True)
            print("Press Ctrl+C to stop. No files are written and no site is published.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        parser.exit(1, f"Could not start local interface: {exc}. Choose another --port if it is in use.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
