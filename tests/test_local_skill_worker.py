"""Run the local HTTP/agent integration with the policy worker's Python and PYTHONPATH."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

executor = pytest.importorskip("synthesis.aspire_service.executor")
from lelab.local_skill_worker import LocalTask, build
from lelab.recovery_skills import SkillLibrary


def test_local_task_api_and_library(tmp_path, monkeypatch):
    for key in ("NORI_S2_SKILLS", "NORI_S2_ADVICE"):
        monkeypatch.setenv(key, "on")
    app = build(tmp_path, "local-token")
    from synthesis import system2_skills
    from synthesis.aspire_service import advice
    assert system2_skills.resolve("index") == ("off", [])
    assert not advice.enabled()

    def simulate(tid, rec, store, context_factory):
        ctx = context_factory(tid, rec, store)
        try:
            def fake_harness(self, instruction):
                return SimpleNamespace(task_text=instruction, system_extra="", extra_tools=[], extra_handler=self.extra,
                                       run_tool=lambda name, args: (args, None), on_event=lambda e: self.emit(e.pop("type"), **e),
                                       model="test", llm_calls=0, _stop_reason=lambda: None)

            with patch.object(executor.TaskContext, "harness", fake_harness):
                h = ctx.harness(rec["instruction"])
            if rec["instruction"] == "retry orientation":
                target = [0.3, -0.15, 0.9]
                # The fake driver echoes its input as the tool result.
                for tool, args in [("move_to", {"xyz": target, "status": "unreachable"}),
                                   ("move_to", {"xyz": target, "fingers": "any", "status": "done"}),
                                   ("hand_pose", {"xyz": target, "status": "ok"})]:
                    h.on_event({"type": "step_start", "tool": tool, "args": args})
                    h.run_tool(tool, args)
            elif not ctx.cons.get("learn", True):
                assert not h.extra_tools and "Recenter" in h.system_extra
            else:
                assert ("Recenter" in h.system_extra) == bool(ctx.candidate)
                for i, (name, args) in enumerate([("gripper", {}), ("verify_hold", {"holding": False}),
                                                   ("move_to", {}), ("gripper", {}), ("verify_hold", {"holding": True})], 1):
                    h.on_event({"type": "step_start", "tool": name, "args": args})
                    result, _ = h.run_tool(name, args)
                    assert result["lab_step"] == i
                    if i == 2:
                        h.extra_handler("mark_step_failure", dict(failed_step=1, observed_step=2, reason="empty jaw"))
                h.extra_handler("submit_recovery", dict(failed_step=1, correction_step=3, evidence_step=5,
                                                       title="Recenter", trigger="empty jaw", correction="move to center"))
            store.update_task(tid, status="done")
        except Exception as exc:
            store.update_task(tid, status="failed", error=repr(exc))
        finally:
            ctx.lt.stop()
            store.unlock(rec["robot"], tid)

    monkeypatch.setattr(executor, "run_task", simulate)
    monkeypatch.setattr(LocalTask, "llm", lambda self: None)
    def review(client, model, proposal, rows, candidate, known_skills):
        return dict(confirmed=True, reason="object rose", scene="box on the left",
                    match_id=candidate["id"] if candidate else None,
                    variation={"before": "center", "after": "left", "step": rows[-1]["step"]} if candidate else None,
                    skill=dict(title="Recenter", trigger="jaw empty", correction="align before closing",
                               limits="before contact", check="object rises", transfer_test="vary the placement"))
    monkeypatch.setattr("lelab.local_skill_worker.assess_recovery", review)

    async def roundtrip():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker/api/v1/aspire/") as c:
            assert (await c.get("robots/mock/notes")).status_code == 401
            c.headers["Authorization"] = "Bearer local-token"
            async def notes():
                return (await c.get("robots/mock/notes")).json()["experiment"]["skills"]

            async def run(constraints, instruction="pick up the box"):
                tid = (await c.post("tasks", json=dict(robot="mock", instruction=instruction,
                                                       constraints=constraints))).json()["task_id"]
                for _ in range(100):
                    rec = (await c.get("tasks/" + tid)).json()
                    if rec["status"] in ("done", "failed"):
                        assert rec["status"] == "done", rec
                        break
                    await asyncio.sleep(.01)
                else:
                    pytest.fail("mock task did not finish")
                return (await c.get(f"tasks/{tid}/events?follow=false")).json()

            assert await notes() == []
            ev = await run({})
            assert ev[-1]["type"] == "skill_queued"
            skill, = await notes()
            assert [e["lab_step"] for e in ev if e["type"] == "step_start"] == [1, 2, 3, 4, 5]
            assert (await run({"verify_skill": skill["id"]}))[-1]["type"] == "skill_verified"
            assert (await notes())[0]["verification"]["task"] != skill["task"]
            await run({"learn": False})
            auto = await run({}, "retry orientation")
            queued, = [e for e in auto if e["type"] == "skill_queued"]
            assert (queued["failed_step"], queued["correction_step"], queued["evidence_step"]) == (1, 2, 3)
            off = await run({"learn": False}, "retry orientation")
            assert not any(e["type"] in {"step_failure", "skill_queued"} for e in off)
            assert not SkillLibrary(tmp_path).skills("another-robot")

    asyncio.run(roundtrip())


def test_reviewer_grounds_skill_and_transfer_in_the_supplied_calls(tmp_path):
    import json
    from lelab.recovery_skills import assess_recovery

    candidate = dict(id="known", task="origin", title="Adapt", trigger="IK failure", correction="Relax orientation",
                     evidence="Box at center", instruction="pick box", calls=[dict(step=2, tool="move_to", args={"xyz": [1, 2, 3]})])
    rows = [dict(step=i, tool=t, args={}, result={}, images=[]) for i, t in enumerate(("move_to", "move_to", "observe"), 1)]
    rows[-1]["images"] = ["observation"]
    proposal = dict(failed_step=1, observed_step=1, correction_step=2, evidence_step=3)
    answer = dict(confirmed=True, reason="Calls 1–3 demonstrate the recovery", match_id="known", scene="box on left",
                  skill=dict(title="Adapt", trigger="IK failure", correction="Relax orientation", limits="before contact",
                             check="Check alignment", transfer_test="Move box"),
                  variation={"before": "center", "after": "left", "step": 3})

    def reply(**kw):
        payload = json.loads(kw["messages"][0]["content"][0]["text"])
        assert payload["skills"][0]["original"]["calls"][0]["args"]["xyz"] == [1, 2, 3]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(answer))])
    client = SimpleNamespace(messages=SimpleNamespace(create=reply))
    assert assess_recovery(client, "test", proposal, rows, candidate)["variation"]["step"] == 3
    answer["variation"]["step"] = 99
    assert assess_recovery(client, "test", proposal, rows, candidate)["variation"] is None
    answer["variation"]["step"] = 2
    assert assess_recovery(client, "test", proposal, rows, candidate)["variation"] is None
    answer.update(variation=None, match_id=None)
    assert not assess_recovery(client, "test", proposal, rows, candidate)["confirmed"]
    answer["match_id"] = "different_robot"
    with pytest.raises(ValueError, match="unknown skill"):
        assess_recovery(client, "test", proposal, rows, candidate)
