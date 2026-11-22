"""Local experimental worker. Run via scripts/local-agent, in the policy-agent environment."""
import asyncio
import json
import os
import secrets
import socket
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from synthesis.aspire_service import executor
from synthesis.aspire_service.api import make_app
from synthesis.aspire_service.grants import grant_for
from synthesis.aspire_service.state import MemDict, Store
from synthesis.real import safety

from lelab.recovery_skills import FAILURE, RECOVERY, RecoveryTrial, SkillLibrary, assess_recovery, skill_wording


class LocalTask(executor.TaskContext):
    def __init__(self, tid, spec, store, *, library, **kwargs):
        robot = spec["robot"]
        self.library = library
        self.recovery_trial = None
        self.skills = library.skills(robot)
        cons = spec.get("constraints") or {}
        self.candidate = None
        if selected := cons.get("verify_skill"):
            self.candidate = next((s for s in self.skills if s["id"] == selected and not s["verification"]), None)
            if not cons.get("learn", True) or not self.candidate:
                raise ValueError("verification requires a pending skill for this robot and learning enabled")
        grant = grant_for(robot)
        if grant.kind == "local_operator":
            if os.environ.get("ASPIRE_ALLOW_HARDWARE") != "1":
                raise PermissionError("start scripts/local-agent with --hardware to use the real robot")
            if not cons.get("operator_presence"):
                raise ValueError("enable stop on close for the local robot's presence heartbeat")
            from synthesis.real.bench_session import BenchGrant
            grant = BenchGrant(robot, gateway_frames=True)
            cfg = executor.hosted_safety_config(robot, cons)

            def alive():
                if store.cancel_request(tid):
                    return False, "task cancelled"
                if store.operator_lost(tid, executor.PRESENCE_TIMEOUT_S):
                    return False, "operator presence expired"
                return True, ""

            def log_safety(event):
                if not event["ok"] or event["check"] in ("session", "single_step"):
                    store.append_event(tid, dict(event, type="safety"))

            kwargs["guard"] = safety.build(cfg, alive=alive, on_event=log_safety)
        kwargs["grant"] = grant
        super().__init__(tid, spec, store, **kwargs)
        self.stereo_only = self.robot_name.upper() == "NORI-A3-0003"
        if self.stereo_only and self.guard and "table_z" not in self.settings:
            self.guard.cfg.table_z = None  # Request the executor's pre-motion support measurement.

    def warm_depth_service(self):
        if not self.stereo_only:
            super().warm_depth_service()

    def open(self):
        super().open()
        if self.stereo_only:
            from lelab.agent_stitch import AgentStitch
            AgentStitch(self.robot_name).attach(self.t, getattr(self.guard, "cfg", None))
        return self

    def emit(self, typ, **data):
        if typ == "continued" and self.recovery_trial is not None:
            self.recovery_trial.begin_instruction(data["instruction"])
        if typ == "table" and data.get("step") == "measuring" and self.stereo_only:
            data["detail"] = "measuring tabletop from overhead and front RGB"
        return super().emit(typ, **data)

    def effective_settings(self):
        settings = super().effective_settings()
        if self.stereo_only:
            settings.update(depth_mode="stereo_triangulation", tof=False)
        return settings

    def harness(self, instruction, **kwargs):
        h = super().harness(instruction, **kwargs)
        if self.stereo_only:
            h.system_extra += ("\nMetric depth comes from features matched in both the front and overhead images. "
                               "Wrist views provide RGB. When a target lacks stereo depth, check its visibility "
                               "in both head views. Supply measured_height_m only for an operator's ruler measurement.")
        verified = [{k: s[k] for k in ("title", "trigger", "correction", "limits", "check", "evidence") if k in s}
                    for s in self.skills if s["verification"]]
        if verified:
            h.system_extra += "\nVerified local recoveries for this robot:\n" + json.dumps(verified)
        if not self.cons.get("learn", True):
            return h

        h.system_extra += ("\nTrack failures per action. Each primitive result includes lab_step. When an action "
                           "misses its subgoal, immediately call mark_step_failure, citing the action and the "
                           "observation establishing the miss. Continue working. After a correction, check that "
                           "the subgoal was achieved and call submit_recovery before moving on. A failed attempt "
                           "or a completed command alone does not establish success. An unreachable move_to is "
                           "marked automatically. After an orientation retry, observe the hand and target.")
        if self.candidate:
            candidate = dict(skill_wording(self.candidate), original_scene=self.candidate.get("scene"),
                             original_evidence=self.candidate["evidence"])
            h.system_extra += ("\nVerification trial: only submit this recovery if its trigger occurs naturally "
                               "and the correction is actually tested. Observe and record what differs from the original "
                               "object, placement or scene and check the outcome. " + json.dumps(candidate))
            self.emit("skill_trial", title=candidate["title"], skill_id=self.candidate["id"])

        def judge(proposal, rows, candidate):
            if h._stop_reason():
                raise ValueError("run budget or stop reached before recovery review")
            h.llm_calls += 1
            try:
                known = [] if candidate else self.library.skills(self.robot_name)
                proof = assess_recovery(self.llm(), h.model, proposal, rows, candidate, known)
            except Exception as exc:
                proof = {"confirmed": False, "reason": f"Recovery review unavailable ({type(exc).__name__})"}
            if self.stopping() or self.store.cancel_request(self.task_id):
                return {"confirmed": False, "reason": "task stopped during recovery review"}
            return proof

        trial = RecoveryTrial(self.library, self.robot_name, "local-" + self.task_id, instruction, self.emit, judge, self.candidate)
        self.recovery_trial = trial
        run_tool, extra = h.run_tool, h.extra_handler
        numbered_call = False

        def record(name, args):
            nonlocal numbered_call
            track_moves, numbered_call = numbered_call, False
            try:
                payload, images = run_tool(name, args)
            except Exception as exc:
                step = trial.record(name, args, {"error": str(exc)}, None)
                raise RuntimeError(f"lab_step {step}: {exc}") from exc
            step = trial.record(name, args, payload, images, default_side=getattr(self.t, "side", "right"),
                                track_moves=track_moves)
            return {"lab_step": step, **(payload if isinstance(payload, dict) else {"result": payload})}, images

        def handle(name, args):
            if name in (FAILURE["name"], RECOVERY["name"]):
                trial.instruction = h.task_text
                return trial.call(name, args), None
            return extra(name, args)

        # Keep the executor's guarded dispatch around every physical command.
        h.run_tool = record
        h.extra_tools.extend((FAILURE, RECOVERY))
        h.extra_handler = handle
        original_event = h.on_event

        def number_step(event):
            nonlocal numbered_call
            if event["type"] == "step_start":
                event["lab_step"] = len(trial.steps) + 1
                numbered_call = True  # The verdict's internal observation has no transcript call number.
            return original_event(event)

        h.on_event = number_step
        return h


class LocalStore(Store):
    """The hardware watcher and the agent may publish events at the same time."""
    def __init__(self):
        super().__init__(MemDict())
        self.event_lock = threading.Lock()

    def append_event(self, tid, event):
        with self.event_lock:
            return super().append_event(tid, event)


def build(folder, token):
    # These settings only apply to this dedicated process, including end-of-task refinement.
    os.environ.update(NORI_S2_SKILLS="off", NORI_S2_ADVICE="0")
    library = SkillLibrary(folder)
    store = LocalStore()
    workers = {}

    def spawn(tid, rec):
        if rec["mode"] != "task":
            raise ValueError("local recovery trials require task mode")
        context = partial(LocalTask, library=library)
        def run():
            try:
                executor.run_task(tid, rec, store, context_factory=context)
            except Exception as exc:
                reason = f"Local worker failed: {type(exc).__name__}: {str(exc)[:200]}"
                store.append_event(tid, {"type": "failed", "reason": reason})
                store.update_task(tid, status="failed", outcome={"reason": reason})
                store.unlock(rec["robot"], tid)
            finally:
                workers.pop(tid, None)

        thread = threading.Thread(target=run, name=f"local-{tid}", daemon=True)
        workers[tid] = thread
        thread.start()

    api = make_app(store, token=token, spawn=spawn, observe_fn=None)

    @api.get("/robots/{robot}/notes")
    def notes(robot: str, authorization: str = Header("")):
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401)
        return {"robot": robot, "files": [], "promotable": [], "lint": [],
                "experiment": {"folder": str(library.folder), "skills": library.skills(robot)}}

    @asynccontextmanager
    async def lifespan(_app):
        yield
        active = list(workers.items())
        for tid, _ in active:
            store.request_cancel(tid, "local worker shutting down")
        await asyncio.gather(*(asyncio.to_thread(thread.join, 10) for _, thread in active))

    app = FastAPI(lifespan=lifespan)
    app.mount("/api/v1/aspire", api)
    return app


def main():
    import inspect
    if "context_factory" not in inspect.signature(executor.run_task).parameters:
        raise SystemExit("Use the nori-policy-agent checkout with the run_task context_factory extension.")
    token = secrets.token_urlsafe(32)
    app = build(os.environ["NORI_LAB_SKILL_DIR"], token)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        connection = Path(os.environ["NORI_LAB_WORKER_CONNECTION"])
        tmp = connection.with_suffix(".tmp")
        tmp.write_text(json.dumps({"port": sock.getsockname()[1], "token": token}))
        tmp.replace(connection)
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])


if __name__ == "__main__":
    main()
