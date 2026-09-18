#!/usr/bin/env python3
"""Serve a baked world's viewer with zoom baking behind it.

    python scripts/serve_world.py --world worlds/earth-v9            # http://localhost:8765/viewer/index.html
    python scripts/serve_world.py --world worlds/earth-v9 --host 100.114.208.82   # this machine's tailnet address only

Static files come from the world directory (the globe viewer, the zoom
pages).  The viewer finds ``/api/zoom`` and turns a click on a spot into a
"Bake a zoom here" button:

* ``GET /api/zoom`` -- ``{"jobs": [...], "zooms": [...]}``; a running job
  carries ``progress`` (globe/zoom/progress.py: fraction, the level running,
  seconds elapsed and an estimate of those left);
* ``POST /api/zoom`` with ``{"face", "i", "j"}`` (a coarse cell) or
  ``{"lat", "lon"}``, and ``"game": true`` for the game-scale levels (19 m,
  4.8 m) below the zoom's -- queues ``scripts/zoom_bake.py`` for that spot;
* ``POST /api/zoom/cancel`` with ``{"name"}`` -- stops a queued or running job
  (a re-bake of the spot resumes from the levels it finished);
* ``GET /api/zoom/log?name=...`` -- the last lines of a job's log.

One bake runs at a time (a zoom uses every core); the rest wait in order.
Each job's log is ``<world>/zoom/<name>/bake.log``; the list is kept in
``<world>/zoom/jobs.json``, so it survives a restart (a job that was running
when the server stopped is marked ``interrupted``).  Listens on localhost
unless ``--host`` says otherwise (a Tailscale address: the tailnet only):
this runs code on the machine for whoever can reach the port.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
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
        self.path = root / "zoom" / "jobs.json"
        self.jobs: list[dict] = self._load()
        self.procs: dict[str, subprocess.Popen] = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def _load(self) -> list[dict]:
        try:
            jobs = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        for j in jobs:
            if j.get("state") == "running":
                j["state"] = "interrupted"
                j["message"] = "the server stopped while it ran; bake again to resume from the levels it finished"
        return jobs

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.jobs, indent=1))
        tmp.replace(self.path)

    def submit(self, spot: tuple[int, int, int], game: bool = False) -> dict:
        name = f"f{spot[0]}_{spot[1]}_{spot[2]}" + ("_game" if game else "")
        with self.lock:
            for j in self.jobs:
                if j["name"] == name and j["state"] in ("queued", "running"):
                    return j
            job = {"name": name, "spot": list(spot), "game": bool(game), "state": "queued", "message": "", "submitted": time.time()}
            self.jobs.append(job)
            self._save()
        return job

    def cancel(self, name: str) -> dict | None:
        with self.lock:
            job = next((j for j in reversed(self.jobs) if j["name"] == name and j["state"] in ("queued", "running")), None)
            if job is None:
                return None
            if job["state"] == "queued":
                job["state"], job["message"] = "cancelled", "cancelled before it started"
                self._save()
                return job
            proc = self.procs.get(name)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)          # the bake and its tile workers
            except ProcessLookupError:
                pass
        with self.lock:
            job["cancel"] = True
        return job

    def snapshot(self) -> list[dict]:
        from globe.zoom import progress

        with self.lock:
            out = [dict(j) for j in self.jobs]
        for j in out:
            if j["state"] in ("running", "interrupted", "cancelled", "failed"):
                j["progress"] = progress.read(self.root / "zoom" / j["name"])
        return out

    def log_tail(self, name: str, lines: int = 60) -> str:
        path = self.root / "zoom" / Path(name).name / "bake.log"
        try:
            return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def _worker(self):
        while True:
            job = None
            with self.lock:
                for j in self.jobs:
                    if j["state"] == "queued":
                        job = j
                        j["state"] = "running"
                        j["started"] = time.time()
                        self._save()
                        break
            if job is None:
                time.sleep(1.0)
                continue
            out = self.root / "zoom" / job["name"]
            out.mkdir(parents=True, exist_ok=True)
            cmd = [self.python, str(HERE / "zoom_bake.py"), "--world", str(self.root), "--cell", *map(str, job["spot"]),
                   "--name", job["name"], *(["--game"] if job.get("game") else []), *self.extra]
            with open(out / "bake.log", "a") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with self.lock:
                    self.procs[job["name"]] = proc
                    job["pid"] = proc.pid
                while proc.poll() is None:
                    time.sleep(2.0)
                    last = _last_line(out / "bake.log")
                    with self.lock:
                        job["message"] = last
            with self.lock:
                self.procs.pop(job["name"], None)
                if job.pop("cancel", False):
                    job["state"] = "cancelled"
                else:
                    job["state"] = "done" if proc.returncode == 0 else "failed"
                job["message"] = _last_line(out / "bake.log")
                job["seconds"] = round(time.time() - job["started"], 1)
                self._save()


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
            # The page and its lists change whenever a bake or an export runs, and a browser
            # holding yesterday's copy looks like the work never happened (a phone kept
            # showing the viewer's old layout for a day).  "no-cache" is not "no-store": the
            # big textures and tiles stay in the cache, but every request revalidates, so a
            # file that has not changed comes back as a 304 (SimpleHTTPRequestHandler answers
            # If-Modified-Since itself)
            path = self.path.split("?")[0]
            if path.endswith("zooms.js"):
                self.send_header("Cache-Control", "no-store")
            elif path.endswith((".html", ".js", ".json")) or path.endswith("/"):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/api/zoom":
                return self._json(200, {"jobs": jobs.snapshot(), "zooms": index.scan(root)})
            if path == "/api/zoom/log":
                from urllib.parse import parse_qs

                name = (parse_qs(query).get("name") or [""])[0]
                return self._json(200, {"name": name, "log": jobs.log_tail(name)})
            return super().do_GET()

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/api/zoom", "/api/zoom/cancel"):
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/zoom/cancel":
                    job = jobs.cancel(str(d["name"]))
                    return self._json(200, job) if job is not None else self._json(404, {"error": "no queued or running job of that name"})
                if "face" in d:
                    spot = (int(d["face"]), int(d["i"]), int(d["j"]))
                else:
                    spot = spot_of_lonlat(params, float(d["lat"]), float(d["lon"]))
                if not (0 <= spot[0] < 6 and 0 <= spot[1] < N and 0 <= spot[2] < N):
                    raise ValueError(f"cell {spot} is not on the grid")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, jobs.submit(spot, bool(d.get("game", False))))

        def log_message(self, fmt, *args):
            # log_error passes an HTTPStatus first, not the request line: taking "in" of it
            # raised inside the handler, which dropped the connection -- a browser asking for
            # /favicon.ico (every one does) left the whole page hanging on an empty reply
            if "/api/" in str(args[0] if args else ""):
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
