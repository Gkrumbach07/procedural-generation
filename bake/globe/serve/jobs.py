"""A queue of bakes, one at a time.

A bake uses every core, so two at once take longer than one after the other:
:class:`JobQueue` runs one and leaves the rest queued, in the order they were
asked for.  A job is a dict -- ``{id, kind, world, title, state, ...}`` --
kept in a JSON file beside the worlds, so the queue survives a restart of the
server (a job that was running when it stopped is marked ``interrupted``, and
a bake resumes from what it finished).

The command a job runs comes from the caller (:func:`JobQueue.submit` takes
it), which keeps this module free of what is being baked; its output goes to
the job's own log, which the page tails.  A job runs in a session of its own,
so cancelling kills the bake *and* the workers it spawned.

States: ``queued`` -> ``running`` -> ``done`` | ``failed`` | ``cancelled``,
or ``interrupted`` when the server stopped under it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

#: how often the worker looks for a new job, and updates a running one's line
POLL_S = 1.0
TICK_S = 2.0
#: states that are still going (a page polls faster while one is)
LIVE = ("queued", "running")


def last_line(path: Path, width: int = 160) -> str:
    """The last line with anything on it, for a job's one-line message."""
    try:
        lines = [x for x in Path(path).read_text(errors="replace").splitlines() if x.strip()]
    except OSError:
        return ""
    return lines[-1][-width:] if lines else ""


def tail(path: Path, lines: int = 200) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class JobQueue:
    """Jobs in ``<store>/jobs.json``, run one at a time by a worker thread."""

    def __init__(self, store: Path, log_dir=None):
        self.path = Path(store)
        self.dir = self.path.parent
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self._log_dir = log_dir
        self.jobs: list[dict] = self._load()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- storage ----------------------------------------------------------
    def _load(self) -> list[dict]:
        try:
            jobs = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        for j in jobs:
            if j.get("state") == "running":
                j["state"] = "interrupted"
                j["message"] = "the server stopped while it ran; run it again to carry on from what it finished"
        return jobs

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.jobs, indent=1))
        tmp.replace(self.path)

    def log_path(self, job: dict) -> Path:
        """Where a job's output goes: its own ``log`` where it named one (a
        zoom keeps its log beside the window it bakes), else ours."""
        if job.get("log"):
            return Path(job["log"])
        if self._log_dir is not None:
            return Path(self._log_dir(job))
        return self.dir / "logs" / f"{job['id']}.log"

    # -- the queue --------------------------------------------------------
    def submit(self, kind: str, world: str, cmd: list[str], title: str = "", **extra) -> dict:
        """Queue ``cmd``.  A job of the same kind and title already queued or
        running is returned instead of a second one."""
        with self.lock:
            for j in self.jobs:
                if j["kind"] == kind and j["world"] == world and j.get("title") == title and j["state"] in LIVE:
                    return dict(j)
            job = {"id": f"{int(time.time() * 1000):x}", "kind": kind, "world": world, "title": title or kind,
                   "cmd": [str(c) for c in cmd], "state": "queued", "message": "", "submitted": time.time(), **extra}
            self.jobs.append(job)
            self._save()
            return dict(job)

    def cancel(self, job_id: str) -> dict | None:
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if job is None:
                return None
            if job["state"] == "queued":
                job["state"] = "cancelled"
                job["message"] = "cancelled before it started"
                self._save()
                return dict(job)
            proc = self.procs.get(job_id)
            if job["state"] == "running" and proc is not None:
                job["cancel"] = True
                job["message"] = "stopping…"
                try:
                    os.killpg(proc.pid, signal.SIGTERM)      # the bake and its workers
                except (ProcessLookupError, PermissionError):
                    pass
            return dict(job)

    def forget(self, job_id: str) -> bool:
        """Drop a finished job from the list (a running one stays)."""
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if job is None or job["state"] in LIVE:
                return False
            self.jobs.remove(job)
            self._save()
            return True

    def snapshot(self, world: str | None = None, progress=None) -> list[dict]:
        """The jobs, newest last; ``progress(job)`` may add a ``progress`` key."""
        with self.lock:
            out = [dict(j) for j in self.jobs if world in (None, j["world"])]
        for j in out:
            j.pop("cmd", None)
            if progress is not None and j["state"] in ("running", "interrupted", "failed", "cancelled"):
                try:
                    j["progress"] = progress(j)
                except (OSError, ValueError, KeyError):
                    j["progress"] = None
        return out

    def log(self, job_id: str, lines: int = 200) -> str:
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
        return tail(self.log_path(job), lines) if job else ""

    def busy(self) -> bool:
        with self.lock:
            return any(j["state"] in LIVE for j in self.jobs)

    # -- the worker -------------------------------------------------------
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
                time.sleep(POLL_S)
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        log = self.log_path(job)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job['cmd'])}\n")
            fh.flush()
            try:
                proc = subprocess.Popen(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                with self.lock:
                    job["state"] = "failed"
                    job["message"] = str(e)
                    self._save()
                return
            with self.lock:
                self.procs[job["id"]] = proc
                job["pid"] = proc.pid
                self._save()
            while proc.poll() is None:
                time.sleep(TICK_S)
                line = last_line(log)
                with self.lock:
                    job["message"] = line
        with self.lock:
            self.procs.pop(job["id"], None)
            if job.pop("cancel", False):
                job["state"] = "cancelled"
            else:
                job["state"] = "done" if proc.returncode == 0 else "failed"
            job["message"] = last_line(log)
            job["seconds"] = round(time.time() - job.get("started", time.time()), 1)
            self._save()


__all__ = ["POLL_S", "TICK_S", "LIVE", "last_line", "tail", "JobQueue"]
