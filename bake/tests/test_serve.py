"""The planet page: the queue behind it, what it reads of a worlds directory,
and the API ``scripts/serve_planets.py`` serves."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from globe.config import STAGES, WorldParams
from globe.serve import hub
from globe.serve import jobs as jb
from globe.serve import worlds as wd


def _wait(fn, timeout=30.0, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return None


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------
def test_the_queue_runs_one_job_at_a_time_in_order(tmp_path):
    """Two jobs asked for at once run one after the other -- a bake takes
    every core -- and each keeps its own log and its seconds."""
    q = jb.JobQueue(tmp_path / "jobs.json")
    a = q.submit("world", "w1", _py("import time; print('A in'); time.sleep(0.6); print('A out')"), title="a")
    b = q.submit("world", "w2", _py("print('B')"), title="b")
    assert [j["state"] for j in q.snapshot()] == ["queued", "queued"]
    assert _wait(lambda: all(j["state"] == "done" for j in q.snapshot()), 40)
    la, lb = q.log(a["id"]), q.log(b["id"])
    assert "A in" in la and "A out" in la and "B" in lb
    done = {j["title"]: j for j in q.snapshot()}
    assert done["a"]["seconds"] >= 0.5 and done["b"]["seconds"] < done["a"]["seconds"] + 5
    assert not q.busy()


def test_a_job_is_not_queued_twice_and_can_be_cancelled(tmp_path):
    q = jb.JobQueue(tmp_path / "jobs.json")
    one = q.submit("zoom", "w", _py("import time; time.sleep(30)"), title="same")
    again = q.submit("zoom", "w", _py("import time; time.sleep(30)"), title="same")
    assert again["id"] == one["id"] and len(q.snapshot()) == 1
    assert _wait(lambda: q.snapshot()[0]["state"] == "running")
    q.cancel(one["id"])
    assert _wait(lambda: q.snapshot()[0]["state"] == "cancelled", 20)
    assert q.forget(one["id"]) and q.snapshot() == []


def test_a_job_the_server_died_under_comes_back_interrupted(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([{"id": "x", "kind": "world", "world": "w", "title": "t",
                                 "cmd": ["true"], "state": "running", "message": ""}]))
    q = jb.JobQueue(path)
    j = q.snapshot()[0]
    assert j["state"] == "interrupted" and "carry on" in j["message"]
    assert q.forget("x") and q.snapshot() == []           # it is not running: it can be cleared


# ---------------------------------------------------------------------------
# the worlds directory
# ---------------------------------------------------------------------------
def test_a_new_world_keeps_the_knobs_it_was_made_with(tmp_path):
    root = wd.create(tmp_path, "w1", "tiny", {"tectonics.steps": 200, "climate.T_eq": 21.0})
    assert (root / wd.PARAMS_NAME).exists()
    p = wd.params_of(root)
    assert p.tectonics.steps == 200 and p.climate.T_eq == pytest.approx(21.0)
    assert p.world.N_c == WorldParams.tiny_world().world.N_c        # the preset's, where no knob said otherwise
    with pytest.raises(FileExistsError):
        wd.create(tmp_path, "w1", "tiny", {})
    for bad in ("", "../escape", "a name"):
        with pytest.raises(ValueError):
            wd.create(tmp_path, bad, "tiny", {})


def test_knobs_are_checked_against_what_they_mean(tmp_path):
    p = WorldParams.tiny_world()
    assert wd.apply(p, {"world.seed": "7"}).world.seed == 7          # a page sends strings
    assert wd.apply(p, {}).to_dict() == p.to_dict()
    for bad in ({"erosion.dt": 2.0}, {"world.N_c": 999}, {"tectonics.steps": -1}, {"climate.T_eq": 500}):
        with pytest.raises(ValueError):
            wd.apply(p, bad)
    assert p.world.seed == WorldParams.tiny_world().world.seed       # the original is untouched


def test_scan_reads_what_a_world_has_baked(tmp_path):
    from globe.pipeline import bake

    root = wd.create(tmp_path, "w1", "tiny", {})
    assert wd.scan(tmp_path)[0]["baked"] is False                     # made, nothing run
    bake(root, wd.params_of(root), to_stage="tectonics", logger=lambda m: None)
    rec = wd.scan(tmp_path)[0]
    assert rec["name"] == "w1" and rec["baked"] and rec["done"] == ["tectonics"] and rec["next"] == "climate"
    assert rec["stages"]["tectonics"]["seconds"] > 0 and rec["stages"]["climate"]["done"] is False
    assert "quicklook/tectonics.png" in rec["quicklooks"] and rec["seed"] == wd.params_of(root).world.seed
    # a world baked from the command line has no params.yaml until the page needs one
    (root / wd.PARAMS_NAME).unlink()
    assert wd.scan(tmp_path)[0]["params_file"] is False
    assert wd.ensure_params(root).exists() and wd.params_of(root).world.N_c == rec["N_c"]


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------
@pytest.fixture()
def server(tmp_path):
    q = jb.JobQueue(tmp_path / "jobs.json")
    handler = partial(hub.make_handler(tmp_path, q, sys.executable), directory=str(tmp_path))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}", "queue": q, "dir": tmp_path}
    srv.shutdown()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _post(url: str, obj: dict):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_the_page_and_its_list_are_served(server):
    page = urllib.request.urlopen(server["url"] + "/", timeout=20).read().decode()
    assert "<title>Planets</title>" in page and "/api/worlds" in page
    d = _get(server["url"] + "/api/worlds")
    assert d["worlds"] == [] and d["stages"] == list(STAGES) and d["busy"] is False
    assert {k["name"] for k in d["knobs"]} >= {"world.seed", "tectonics.steps", "erosion.iterations"}


def test_a_world_is_made_and_its_stage_baked_through_the_api(server):
    made = _post(server["url"] + "/api/worlds", {"name": "p1", "preset": "tiny", "sets": {"tectonics.steps": 150}, "to": "tectonics"})
    assert made["created"] and made["job"]["state"] in jb.LIVE
    root = server["dir"] / "p1"
    assert wd.params_of(root).tectonics.steps == 150
    job = _wait(lambda: next((j for j in _get(server["url"] + "/api/worlds")["jobs"] if j["state"] not in jb.LIVE), None), 180)
    assert job and job["state"] == "done", job
    rec = _get(server["url"] + "/api/worlds")["worlds"][0]
    assert rec["done"] == ["tectonics"] and rec["viewer"] is True
    assert "[tectonics]" in _get(f"{server['url']}/api/log?id={job['id']}")["log"]
    # and its viewer is served under the world's own name, where its API calls land
    assert urllib.request.urlopen(f"{server['url']}/p1/viewer/index.html", timeout=20).status == 200
    assert _get(f"{server['url']}/p1/api/zoom") == {"jobs": [], "zooms": []}


def test_the_api_refuses_what_it_cannot_run(server):
    assert "error" in _post(server["url"] + "/api/worlds", {"name": "../escape", "preset": "tiny"})
    assert "error" in _post(server["url"] + "/api/worlds", {"name": "ok", "preset": "nope"})
    assert "error" in _post(server["url"] + "/api/bake", {"world": "missing"})
    assert "error" in _post(server["url"] + "/api/planet", {"world": "missing", "R": 8})
    _post(server["url"] + "/api/worlds", {"name": "p2", "preset": "tiny"})
    assert "error" in _post(server["url"] + "/api/bake", {"world": "p2", "to": "not-a-stage"})
    assert _get(server["url"] + "/api/worlds")["jobs"] == []


def test_a_bake_runs_with_the_worlds_own_parameters(tmp_path):
    root = wd.create(tmp_path, "w", "tiny", {})
    cmd = hub.bake_cmd("python3", root, "erosion", "hydro", force=True)
    assert cmd[1].endswith("bake.py")
    assert cmd[cmd.index("--world") + 1] == str(root)
    assert cmd[cmd.index("--params") + 1] == str(wd.params_path(root))
    assert cmd[cmd.index("--from") + 1] == "erosion" and cmd[cmd.index("--to") + 1] == "hydro" and "--force" in cmd


def test_progress_comes_from_the_line_a_stage_last_logged(tmp_path):
    q = jb.JobQueue(tmp_path / "jobs.json")
    log = tmp_path / "j.log"
    log.write_text("[   1.0s] [erosion] iter 120/800: 4 particles\n")
    pr = hub.job_progress(tmp_path, {"kind": "world", "world": "w", "id": "j", "log": str(log)})
    assert pr == {"stage": "erosion", "step": 120, "of": 800, "fraction": pytest.approx(0.15)}
    log.write_text("[   1.0s] [tiles] writing\n")
    assert hub.job_progress(tmp_path, {"kind": "world", "world": "w", "id": "j", "log": str(log)}) is None
    assert q.snapshot() == []
