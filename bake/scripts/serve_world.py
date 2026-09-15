#!/usr/bin/env python3
"""Serve a baked world's viewer with zoom baking behind it.

    python scripts/serve_world.py --world worlds/earth-v9            # http://localhost:8765/viewer/index.html
    python scripts/serve_world.py --world worlds/earth-v9 --host 100.114.208.82   # this machine's tailnet address only

Static files come from the world directory (the globe viewer, the zoom
pages).  The viewer finds ``/api/zoom`` and turns a click on a spot into a
"Bake a zoom here" button:

* ``GET /api/zoom`` -- ``{"jobs": [...], "zooms": [...]}``;
* ``POST /api/zoom`` with ``{"face", "i", "j"}`` (a coarse cell) or
  ``{"lat", "lon"}`` -- queues ``scripts/zoom_bake.py`` for that spot.

One bake runs at a time (a zoom uses every core); the rest wait in order.
Each job's log is ``<world>/zoom/<name>/bake.log``.  Listens on localhost
unless ``--host`` says otherwise (a Tailscale address: the tailnet only):
this runs code on the machine for whoever can reach the port.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Jobs:
    def __init__(self, root: Path, python: str, extra: list[str]):
        self.root, self.python, self.extra = root, python, extra
        self.lock = threading.Lock()
        self.jobs: list[dict] = []
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, spot: tuple[int, int, int]) -> dict:
        name = f"f{spot[0]}_{spot[1]}_{spot[2]}"
        with self.lock:
            for j in self.jobs:
                if j["name"] == name and j["state"] in ("queued", "running"):
                    return j
            job = {"name": name, "spot": list(spot), "state": "queued", "message": "", "submitted": time.time()}
            self.jobs.append(job)
        return job

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [dict(j) for j in self.jobs]

    def _worker(self):
        while True:
            job = None
            with self.lock:
                for j in self.jobs:
                    if j["state"] == "queued":
                        job = j
                        j["state"] = "running"
                        j["started"] = time.time()
                        break
            if job is None:
                time.sleep(1.0)
                continue
            out = self.root / "zoom" / job["name"]
            out.mkdir(parents=True, exist_ok=True)
            cmd = [self.python, str(HERE / "zoom_bake.py"), "--world", str(self.root), "--cell", *map(str, job["spot"]), *self.extra]
            with open(out / "bake.log", "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    time.sleep(2.0)
                    last = _last_line(out / "bake.log")
                    with self.lock:
                        job["message"] = last
            with self.lock:
                job["state"] = "done" if proc.returncode == 0 else "failed"
                job["message"] = _last_line(out / "bake.log")
                job["seconds"] = round(time.time() - job["started"], 1)


def _last_line(path: Path) -> str:
    try:
        lines = [x for x in path.read_text(errors="replace").splitlines() if x.strip()]
    except OSError:
        return ""
    return lines[-1][-160:] if lines else ""


def make_handler(root: Path, jobs: Jobs):
    from globe.config import WorldParams
    from globe.io.world_store import WorldStore
    from globe.zoom import index
    from globe.zoom.bake import spot_of_lonlat

    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    N = params.coarse_grid().N

    class Handler(SimpleHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            if self.path.split("?")[0].endswith("zooms.js"):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            if self.path.split("?")[0] == "/api/zoom":
                return self._json(200, {"jobs": jobs.snapshot(), "zooms": index.scan(root)})
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/api/zoom":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n) or b"{}")
                if "face" in d:
                    spot = (int(d["face"]), int(d["i"]), int(d["j"]))
                else:
                    spot = spot_of_lonlat(params, float(d["lat"]), float(d["lon"]))
                if not (0 <= spot[0] < 6 and 0 <= spot[1] < N and 0 <= spot[2] < N):
                    raise ValueError(f"cell {spot} is not on the grid")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, jobs.submit(spot))

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else ""):
                return
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default localhost; e.g. this machine's Tailscale IP)")
    ap.add_argument("--bake-args", nargs=argparse.REMAINDER, default=[], help="extra arguments for zoom_bake.py (e.g. --levels ...)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE.parent))
    root = Path(a.world).resolve()
    if not (root / "manifest.json").exists():
        ap.error(f"{root} is not a baked world")
    from globe.zoom import index

    index.write(root)
    jobs = Jobs(root, sys.executable, a.bake_args)
    server = ThreadingHTTPServer((a.host, a.port), partial(make_handler(root, jobs), directory=str(root)))
    print(f"serving {root} at http://{a.host}:{a.port}/viewer/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
