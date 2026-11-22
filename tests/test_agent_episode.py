import json

from lelab.agent_episode import save_kept_timelines, tool_at_joint_time


def test_kept_episode_recovers_calls_before_last_page_and_matches_joint_samples(tmp_path):
    events = [{"n": n, "type": "frame"} for n in range(1030)]

    def event(n, typ, **fields):
        events[n] = {"n": n, "type": typ, **fields}

    ep = "episode-0002"
    event(2, "episode_done", episode="episode-0001", keep=False)
    event(3, "record", verb="episode_start", ok=True, episode=ep, ts=100.2, ms=400)
    event(4, "episode_started", episode=ep, ts=100.4)
    event(5, "step_start", ts=101, tool="lift", args={"height": 0.9})
    event(502, "step", ts=102, tool="lift", status="ok")
    event(503, "step_start", ts=102, tool="lift", args={"height": 0.7})
    event(510, "step", ts=103, tool="lift", status="ok")
    event(1011, "step", ts=110, tool="pick")  # missing start
    event(1012, "step_start", ts=111, tool="open")  # missing end
    event(1029, "episode_done", episode=ep, keep=True, index=2)
    reads = []

    def read(since):
        reads.append(since)
        return events[since:since + 500]

    save_kept_timelines("task", events[-1:], read, tmp_path)
    assert len(reads) == 3
    dest = tmp_path / "task" / f"{ep}.json"
    timeline = json.loads(dest.read_text())
    assert timeline["start_unix_s"] == 100
    assert timeline["record_start_round_trip_ms"] == 400
    assert len(timeline["calls"]) == 2
    assert tool_at_joint_time(timeline, 101)["args"] == {"height": 0.9}
    assert tool_at_joint_time(timeline, 5002, first_joint_ts=5000)["args"] == {"height": 0.7}
    assert tool_at_joint_time(timeline, 5003, first_joint_ts=5000) is None
    assert tool_at_joint_time(timeline, 115) is None
    save_kept_timelines("task", events[-1:], read, tmp_path)
    assert len(reads) == 3


def test_continued_task_keeps_only_its_own_completed_calls(tmp_path):
    page = [
        {"type": "episode_started", "episode": "first", "ts": 10},
        {"type": "step_start", "tool": "pick", "ts": 11},
        {"type": "episode_done", "episode": "first", "keep": False},
        {"type": "episode_started", "episode": "second", "ts": 30},
        {"type": "step", "tool": "pick", "ts": 31},
        {"type": "episode_done", "episode": "second", "keep": True},
    ]
    page = [dict(ev, n=i) for i, ev in enumerate(page)]
    save_kept_timelines("task", page, None, tmp_path)
    assert not (tmp_path / "task/first.json").exists()
    timeline = json.loads((tmp_path / "task/second.json").read_text())
    assert timeline["start_reference"] == "episode_started_event"
    assert timeline["calls"] == []
