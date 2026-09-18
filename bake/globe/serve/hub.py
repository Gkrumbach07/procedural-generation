"""The planet page: every world in one place, and the bakes behind it.

``scripts/serve_planets.py`` serves a directory of worlds.  The page at ``/``
lists them, makes new ones and runs their stages; each world is served whole
under ``/<name>/``, so its viewer is ``/<name>/viewer/index.html`` and the
zoom API that page asks for (``../api/zoom``) lands back here as
``/<name>/api/zoom``.

The API:

* ``GET  /api/worlds`` -- ``{worlds, knobs, presets, stages, jobs}``: what
  :mod:`globe.serve.worlds` reads plus the queue (:mod:`globe.serve.jobs`);
* ``POST /api/worlds`` ``{name, preset, sets, to}`` -- make a world and, with
  ``to``, queue its stages up to that one;
* ``POST /api/bake`` ``{world, from, to, sets, force}`` -- set knobs and run
  stages; ``sets`` rewrites the world's ``params.yaml`` first;
* ``POST /api/planet`` ``{world, R, iterations}`` -- the planet-wide level at
  ``R`` (``globe/zoom/planet.py``), which keeps the time lapse;
* ``POST /api/viewer`` ``{world}`` -- export the viewer, even half-baked;
* ``POST /api/cancel`` / ``POST /api/forget`` ``{id}``;
* ``GET  /api/log?id=`` -- the job's output;
* ``GET|POST /<world>/api/zoom...`` -- the zoom bakes, as
  ``scripts/serve_world.py`` serves them.

One job runs at a time, whatever kind: a bake takes every core.
"""
from __future__ import annotations

import json
import re
import sys
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

from ..config import STAGES
from . import jobs as jb
from . import worlds as wd

#: the page itself
PAGE = Path(__file__).resolve().parents[1] / "viz" / "hub.html"
#: scripts a job runs
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
#: "[erosion] iter 12/800", "[tectonics] step 1500/3000" -- how far a stage is
_STEP_RE = re.compile(r"\[(\w+)\]\s+(?:iter|step)\s+(\d+)\s*/\s*(\d+)")


def bake_cmd(python: str, root: Path, from_stage: str | None = None, to_stage: str | None = None,
             force: bool = False) -> list[str]:
    """The command that runs a world's stages with its own ``params.yaml``."""
    cmd = [python, str(SCRIPTS / "bake.py"), "--world", str(root), "--params", str(wd.params_path(root))]
    if from_stage:
        cmd += ["--from", from_stage]
    if to_stage:
        cmd += ["--to", to_stage]
    if force:
        cmd += ["--force"]
    return cmd


def job_progress(worlds_dir: Path, job: dict) -> dict | None:
    """How far a job has got: a zoom bake's plan, a planet bake's passes, or
    the stage and iteration a world bake's log last wrote."""
    root = Path(worlds_dir) / job["world"]
    if job["kind"] == "zoom":
        from ..zoom import progress

        return progress.read(root / "zoom" / job["title"])
    line = jb.last_line(Path(job.get("log") or (Path(worlds_dir) / "logs" / f"{job['id']}.log")))
    m = _STEP_RE.search(line or "")
    if m:
        return {"stage": m.group(1), "step": int(m.group(2)), "of": int(m.group(3)),
                "fraction": int(m.group(2)) / max(int(m.group(3)), 1)}
    return None


def make_handler(worlds_dir: Path, queue: jb.JobQueue, python: str = "", extra: list[str] | None = None):
    worlds_dir = Path(worlds_dir)
    python = python or sys.executable
    extra = list(extra or [])

    class Handler(SimpleHTTPRequestHandler):
        # -- plumbing -----------------------------------------------------
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def end_headers(self):
            path = self.path.split("?")[0]
            if path.endswith(("zooms.js", "jobs.json")):
                self.send_header("Cache-Control", "no-store")
            elif path.endswith((".html", ".js", ".json")) or path.endswith("/"):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, fmt, *args):
            # log_error passes an HTTPStatus first, not the request line
            if "/api/" in str(args[0] if args else ""):
                return
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- the world a path is in ---------------------------------------
        def _world(self, name: str) -> Path:
            root = (worlds_dir / Path(name).name).resolve()
            if not str(root).startswith(str(worlds_dir.resolve())) or not root.is_dir():
                raise ValueError(f"{name}: no such world")
            return root

        def _snapshot(self, world: str | None = None) -> list[dict]:
            return queue.snapshot(world, progress=lambda j: job_progress(worlds_dir, j))

        # -- GET ----------------------------------------------------------
        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html", "/hub.html"):
                return self._page()
            if path == "/api/worlds":
                return self._json(200, {"worlds": wd.scan(worlds_dir), "knobs": wd.KNOBS, "stages": list(STAGES),
                                        "presets": ["earth", "default", "small", "tiny"], "jobs": self._snapshot(),
                                        "busy": queue.busy()})
            if path == "/api/log":
                jid = (parse_qs(query).get("id") or [""])[0]
                return self._json(200, {"id": jid, "log": queue.log(jid)})
            m = re.match(r"^/([^/]+)/api/zoom(/log)?$", path)
            if m:
                return self._zoom_get(m.group(1), m.group(2), query)
            return super().do_GET()

        def _page(self):
            try:
                body = PAGE.read_bytes()
            except OSError:
                return self.send_error(500, "the page is missing")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _zoom_get(self, name: str, sub: str | None, query: str):
            from ..zoom import index as zoom_index

            try:
                root = self._world(name)
            except ValueError as e:
                return self._json(404, {"error": str(e)})
            if sub:
                jid = (parse_qs(query).get("name") or [""])[0]
                job = next((j for j in self._snapshot(name) if j.get("title") == jid), None)
                return self._json(200, {"name": jid, "log": queue.log(job["id"]) if job else ""})
            jobs = [{**j, "name": j.get("title")} for j in self._snapshot(name) if j["kind"] == "zoom"]
            return self._json(200, {"jobs": jobs, "zooms": zoom_index.scan(root)})

        # -- POST ---------------------------------------------------------
        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                d = self._body()
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            try:
                if path == "/api/worlds":
                    return self._json(200, self._create(d))
                if path == "/api/bake":
                    return self._json(200, self._bake(d))
                if path == "/api/planet":
                    return self._json(200, self._planet(d))
                if path == "/api/viewer":
                    return self._json(200, self._viewer(d))
                if path == "/api/cancel":
                    return self._json(200, queue.cancel(str(d.get("id"))) or {"error": "no such job"})
                if path == "/api/forget":
                    return self._json(200, {"forgotten": queue.forget(str(d.get("id")))})
                m = re.match(r"^/([^/]+)/api/zoom(/cancel)?$", path)
                if m:
                    return self._json(200, self._zoom_post(m.group(1), m.group(2), d))
            except (ValueError, FileExistsError, FileNotFoundError, KeyError) as e:
                return self._json(400, {"error": f"{type(e).__name__}: {e}"})
            return self._json(404, {"error": "not found"})

        def _create(self, d: dict) -> dict:
            root = wd.create(worlds_dir, str(d.get("name", "")), str(d.get("preset", "earth")), d.get("sets") or {})
            out = {"world": root.name, "created": True}
            if d.get("to"):
                out["job"] = queue.submit("world", root.name, bake_cmd(python, root, to_stage=str(d["to"])) + extra,
                                          title=f"bake to {d['to']}", to=str(d["to"]))
            return out

        def _bake(self, d: dict) -> dict:
            root = self._world(str(d.get("world", "")))
            wd.ensure_params(root)
            sets = d.get("sets") or {}
            if sets:
                wd.write_params(root, wd.apply(wd.params_of(root), sets))
            frm, to = d.get("from") or None, d.get("to") or None
            for s in (frm, to):
                if s is not None and s not in STAGES:
                    raise ValueError(f"{s}: not a stage")
            force = bool(d.get("force")) or bool(sets)
            title = f"bake {frm or STAGES[0]}…{to or STAGES[-1]}"
            return {"job": queue.submit("world", root.name, bake_cmd(python, root, frm, to, force) + extra,
                                        title=title, to=to, **({"sets": sets} if sets else {}))}

        def _planet(self, d: dict) -> dict:
            root = self._world(str(d.get("world", "")))
            R = int(d.get("R") or 8)
            cmd = [python, str(SCRIPTS / "planet_bake.py"), "--world", str(root), "--R", str(R)]
            if d.get("iterations"):
                cmd += ["--iterations", str(int(d["iterations"]))]
            return {"job": queue.submit("planet", root.name, cmd, title=f"planet R={R}", R=R)}

        def _viewer(self, d: dict) -> dict:
            root = self._world(str(d.get("world", "")))
            cmd = [python, str(SCRIPTS / "export_viewer.py"), "--world", str(root)]
            planet = sorted((root / "zoom").glob("planet_R*")) if (root / "zoom").is_dir() else []
            planet = [p for p in planet if (p / "planet.json").exists()]
            if planet:
                cmd += ["--planet", f"zoom/{planet[-1].name}", "--detail"]
            return {"job": queue.submit("viewer", root.name, cmd, title="export viewer")}

        def _zoom_post(self, name: str, sub: str | None, d: dict) -> dict:
            from ..config import WorldParams
            from ..io.world_store import WorldStore
            from ..zoom.bake import spot_of_lonlat

            root = self._world(name)
            if sub:                                  # /cancel, by the zoom's name
                job = next((j for j in self._snapshot(name) if j.get("title") == str(d.get("name"))), None)
                return queue.cancel(job["id"]) if job else {"error": "no such job"}
            params = WorldParams.from_dict(WorldStore(root).manifest["params"])
            N = params.coarse_grid().N
            if "lat" in d and "lon" in d:
                spot = spot_of_lonlat(N, float(d["lat"]), float(d["lon"]))
            else:
                spot = (int(d["face"]), int(d["i"]), int(d["j"]))
            if not (0 <= spot[0] < 6 and 0 <= spot[1] < N and 0 <= spot[2] < N):
                raise ValueError(f"{spot}: not a cell of this world")
            game = bool(d.get("game", False))
            zname = f"f{spot[0]}_{spot[1]}_{spot[2]}" + ("_game" if game else "")
            cmd = [python, str(SCRIPTS / "zoom_bake.py"), "--world", str(root), "--cell", *map(str, spot), "--name", zname]
            if game:
                cmd += ["--game"]
            job = queue.submit("zoom", root.name, cmd + extra, title=zname, spot=list(spot), game=game,
                               log=str(root / "zoom" / zname / "bake.log"))
            return {**job, "name": zname}

    return Handler


__all__ = ["PAGE", "SCRIPTS", "bake_cmd", "job_progress", "make_handler"]
