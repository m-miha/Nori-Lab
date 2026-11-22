"""Local tool timelines for recordings made from the Agent page."""

import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


def tool_at_joint_time(timeline, joint_ts, *, first_joint_ts=None):
    """Find the active call at a raw observations.ndjson timestamp, or None in a gap.

    For independent clocks, pass the first joint timestamp from the same episode.
    This approximates recording start by the first observation. Without it, both
    clocks must already agree on Unix seconds.
    """
    ts = joint_ts if first_joint_ts is None else timeline["start_unix_s"] + (joint_ts - first_joint_ts)
    return next((c for c in timeline["calls"] if c["start_unix_s"] <= ts < c["end_unix_s"]), None)


def _timestamp(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _episode_events(page, fetch_page, done):
    # A local tab may attach halfway through a recording. Walk back through the
    # persisted stream only as far as this take. Drop frame payloads as we go.
    events = []
    before = done["n"]
    while True:
        for ev in reversed(page):
            if ev["n"] >= before:
                continue
            if ev["type"] == "episode_done":
                return reversed(events)
            if ev["type"] == "record" and ev.get("verb") == "episode_start" and ev.get("ok"):
                events.append(ev)
                return reversed(events)
            if ev["type"] in ("episode_started", "step_start", "step"):
                events.append(ev)
        # The API's 'since' is inclusive and a page holds up to 500 events.
        before = page[0]["n"]
        if before <= 0:
            return reversed(events)
        earlier = fetch_page(max(0, before - 500))
        if not earlier or earlier[0]["n"] >= before:
            return reversed(events)
        page = earlier


def _timeline(task_id, done, events):
    start = ack = pending = None
    calls = []
    for ev in events:
        ts = ev.get("ts")
        if ev["type"] == "record" and ev.get("episode") == done["episode"] and _timestamp(ts):
            ms = ev.get("ms")
            if _timestamp(ms) and ms >= 0:
                ack = (ts, ms)
        elif ev["type"] == "episode_started":
            start = ev if ev.get("episode") == done["episode"] and _timestamp(ts) else None
            calls.clear()
            pending = None
        elif start and ev["type"] == "step_start":
            pending = ev if _timestamp(ts) and isinstance(ev.get("tool"), str) else None
        elif start and ev["type"] == "step":
            if pending and ev.get("tool") == pending["tool"] and _timestamp(ts) and ts >= pending["ts"]:
                calls.append({"tool": pending["tool"], "args": pending.get("args", {}),
                              "start_unix_s": pending["ts"], "end_unix_s": ts, "status": ev.get("status")})
            pending = None
    if start is None:
        return None
    # The robot begins capturing during the recording-start request. Its midpoint
    # estimates that instant on the worker clock without relying on browser arrival time.
    return {"task_id": task_id, "episode": done["episode"], "instruction_index": done.get("index"),
            "start_unix_s": ack[0] - ack[1] / 2000 if ack else start["ts"],
            "start_reference": "record_request_midpoint" if ack else "episode_started_event",
            "record_start_round_trip_ms": ack[1] if ack else None, "calls": calls}


def save_kept_timelines(task_id, page, fetch_page, root: Path):
    """Materialize kept episodes when their completion is read through the local proxy."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        return
    for ev in page:
        if ev["type"] != "episode_done" or ev.get("keep") is not True:
            continue
        episode = ev.get("episode")
        if not isinstance(episode, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", episode):
            continue
        directory = root / task_id
        dest = directory / f"{episode}.json"
        if dest.exists():
            continue
        timeline = _timeline(task_id, ev, _episode_events(page, fetch_page, ev))
        if timeline is None:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(dir=directory, delete=False, mode="w", encoding="utf-8")
        try:
            with tmp:
                json.dump(timeline, tmp, allow_nan=False)
            os.replace(tmp.name, dest)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
