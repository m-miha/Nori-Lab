from tempfile import TemporaryDirectory
from unittest import TestCase

from lelab.recovery_skills import FAILURE, RECOVERY, RecoveryTrial, SkillLibrary


def review(confirmed, reason, match=None, variation=None):
    return dict(confirmed=confirmed, reason=reason, match_id=match, variation=variation, scene="Observed target",
                skill=dict(title="Adapt the approach", trigger="Approach missed", correction="Adapt and observe",
                           limits="While clear of contact", check="Check alignment", transfer_test="Move the target"))


class RecoveryTests(TestCase):
    def test_failed_grasp_then_independent_trial(self):
        with TemporaryDirectory() as path:
            db = SkillLibrary(path)
            self.assertEqual(db.skills("1005"), [])
            events = []

            def trial(task, candidate=None):
                def judge(proposal, rows, selected):
                    self.assertEqual(selected, candidate)
                    holding = rows[-1]["result"]["holding"]
                    return review(holding, f"verify_hold holding={holding}", candidate["id"] if candidate else None,
                                  {"before": "box at center", "after": "box on left", "step": rows[-1]["step"]} if candidate else None)
                run = RecoveryTrial(db, "1005", task, "pick up the box", lambda t, **kw: events.append((t, kw)),
                                    judge, candidate)
                for name, result in [("gripper", {}), ("verify_hold", {"holding": False}),
                                     ("locate", {}), ("move_to", {}), ("gripper", {})]:
                    run.record(name, {}, result, None)
                run.call(FAILURE["name"], {"failed_step": 1, "observed_step": 2, "reason": "empty jaw"})
                return run

            proposal = dict(failed_step=1, correction_step=3, evidence_step=6,
                            title="recenter", trigger="empty jaw", correction="relocate before closing")
            origin = trial("original")
            with self.assertRaises(ValueError):
                origin.call(RECOVERY["name"], proposal)
            origin.record("verify_hold", {}, {"holding": True}, None)
            queued = origin.call(RECOVERY["name"], proposal)
            self.assertEqual(origin.call(RECOVERY["name"], proposal), queued)
            self.assertEqual([e[0] for e in events], ["step_failure", "skill_queued"])
            self.assertEqual(db.skills("other"), [])
            candidate, = SkillLibrary(path).skills("1005")
            self.assertIsNone(candidate["verification"])
            self.assertEqual(candidate["title"], "Adapt the approach")
            self.assertEqual(candidate["suggestion"]["title"], "recenter")

            retry = trial("fresh", candidate)
            retry.record("verify_hold", {}, {"holding": False}, None)
            self.assertEqual(retry.call(RECOVERY["name"], proposal)["status"], "unconfirmed")
            self.assertIsNone(db.skills("1005")[0]["verification"])
            retry.record("verify_hold", {}, {"holding": True}, None)
            retry.call(RECOVERY["name"], dict(proposal, evidence_step=7))
            verified, = SkillLibrary(path).skills("1005")
            self.assertEqual(verified["verification"]["task"], "fresh")
            self.assertEqual(verified["verification"]["evidence_step"], 7)
            self.assertEqual(events[-1][0], "skill_verified")

    def test_orientation_retry_is_reviewed_after_evidence(self):
        with TemporaryDirectory() as path:
            db = SkillLibrary(path)
            events, reviews = [], []

            def judge(proposal, rows, candidate):
                reviews.append(proposal["evidence_step"])
                self.assertEqual(rows[0]["tool"], "move_to")
                self.assertEqual(rows[-1]["images"], ["frame"])
                return review(len(reviews) == 2, "reviewed the observed hand and target")

            run = RecoveryTrial(db, "1005", "run", "pick", lambda t, **kw: events.append(t), judge)
            target = {"xyz": [0.307, -0.156, 0.90]}
            run.record("move_to", target, {"status": "unreachable", "reason": "no IK"}, None)
            run.record("move_to", {"xyz": [0.22, -0.16, 0.82]}, {"status": "unreachable"}, None)
            run.record("move_to", dict(target, fingers="any", jaw_yaw_deg=90), {"status": "done"}, None)
            self.assertEqual(events, ["step_failure", "step_failure"])
            run.record("observe", {}, {"status": "ok"}, ["frame"])
            self.assertEqual(reviews, [4])
            self.assertEqual(db.skills("1005"), [])
            run.record("observe", {}, {"status": "ok"}, ["frame"])
            run.record("gripper", {"command": "close"}, {"status": "done"}, None)
            run.record("observe", {}, {"status": "ok"}, ["frame"])
            skill, = db.skills("1005")
            self.assertEqual(reviews, [4, 7])
            self.assertEqual((skill["failed_step"], skill["correction_step"], skill["evidence_step"]), (1, 3, 7))
            self.assertIsNone(skill["verification"])
            self.assertEqual(run.call(RECOVERY["name"], {"failed_step": 1})["skill_id"], skill["id"])
            self.assertEqual(events.count("skill_queued"), 1)

    def test_unrelated_moves_and_observations_do_not_trigger_review(self):
        with TemporaryDirectory() as path:
            reviews = []
            run = RecoveryTrial(SkillLibrary(path), "1005", "run", "pick", lambda *a, **k: None,
                                lambda *a: reviews.append(a) or {"confirmed": False, "reason": "unexpected review"})
            xyz = [0.3, -0.15, 0.9]
            def move(status, **args):
                run.record("move_to", {"xyz": xyz, **args}, {"status": status}, None)
            def observe():
                run.record("observe", {}, {"status": "ok"}, ["frame"])
            move("unreachable", arm="left")
            move("done", side="right", fingers="any")
            observe()
            move("done", side="left", xyz=[0.3, -0.15, 0.8], fingers="any")
            observe()
            move("done", side="left", fingers="down")
            observe()
            move("unreachable")
            move("stopped_short", fingers="any")
            observe()
            move("done", fingers="any")
            run.record("hand_pose", {"side": "left"}, {"status": "ok", "xyz": xyz}, None)
            run.record("observe", {}, {"status": "ok"}, ["frame"], track_moves=False)
            run.record("observe", {}, {"status": "ok"}, None)
            run.begin_instruction("another task")
            observe()
            self.assertEqual(reviews, [])

    def test_match_adds_examples_and_only_a_demonstrated_transfer_verifies(self):
        with TemporaryDirectory() as path:
            db = SkillLibrary(path)
            original = dict(title="Concrete retry", trigger="no IK", correction="relax wrist", instruction="pick box",
                            failed_step=1, observed_step=1, correction_step=2, evidence_step=3,
                            evidence="Original black box", calls=[{"step": 1, "tool": "move_to"}])
            key = db.queue("1005", "original", original)  # A pending entry created by the earlier worker.
            proof = review(True, "Fresh outcome", key)
            example = dict(original, **proof["skill"], scene="box in center")
            for _ in range(2):
                event, skill_id, _ = db.save_review("1005", "second", example, proof)
                self.assertEqual((event, skill_id), ("skill_supported", key))
            self.assertEqual(db.save_review("1005", "repeat", example, proof, key)[0], "skill_supported")
            skill, = db.skills("1005")
            self.assertIsNone(skill["verification"])
            self.assertEqual(skill["calls"], original["calls"])
            self.assertEqual({e["task"] for e in skill["examples"]}, {"second", "repeat"})
            with self.assertRaises(ValueError):
                db.save_review("0003", "other", example, proof)
            with self.assertRaises(ValueError):
                db.save_review("1005", "original", dict(example, failed_step=4), proof, key)
            proof["variation"] = {"before": "center", "after": "left", "step": 3}
            event, _, _ = db.save_review("1005", "transfer", example, proof, key)
            self.assertEqual(event, "skill_verified")
            skill, = db.skills("1005")
            self.assertEqual(skill["verification"]["variation"], proof["variation"])
            self.assertEqual(len(skill["examples"]), 3)
            db.save_review("1005", "later", dict(example, title="Broader wording"), dict(proof, variation=None))
            self.assertEqual(db.skills("1005")[0]["title"], skill["title"])
            db.save_review("1005", "new_setup", dict(example, title="Broader wording"), proof)
            self.assertEqual(db.skills("1005")[0]["title"], "Broader wording")
