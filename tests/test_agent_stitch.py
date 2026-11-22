"""Exercise stereo through the same observe and locate entry points as the agent."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("synthesis.real.stitch_map")
import cv2
from lelab.agent_stitch import AgentStitch
from synthesis.perception.primitives import Perception
from synthesis.perception.real_body import RealBody
from synthesis.real import robot_profile, segment, stitch_map, support, world
from synthesis.real.safety import SafetyConfig
from synthesis.real.task import RealA3Task


def test_observe_and_locate_use_only_current_stereo(monkeypatch):
    monkeypatch.setenv("NORI_CAMERA_PRIOR", "1")
    robot_profile.activate("NORI-A3-0003")
    loop = asyncio.new_event_loop()
    try:
        task = RealA3Task(SimpleNamespace(telemetry=SimpleNamespace(state={"lift.pos": 450})), loop=loop)
        stereo = AgentStitch("NORI-A3-0003")
        stereo.attach(task, SafetyConfig(table_z=None))
        size = (320, 240)
        # Render pinhole cameras. Lens correction is identity for this synthetic rig.
        stereo.cursor.calib.update(dist_a=np.zeros(5), dist_b=np.zeros(5))
        rng = np.random.default_rng(13)
        def texture():
            return cv2.resize(rng.integers(0, 255, (120, 160), dtype=np.uint8), size)
        table, box = texture(), texture()
        mask = np.zeros_like(table)
        mask[52:98, 140:185] = 255
        frames = {}
        mats = [stitch_map._Kmat(stereo.intrinsics(r, size)[0]) for r in stereo.roles]
        poses = [stereo.pose(r, 0.45) for r in stereo.roles]
        def warp(img, z):
            H = stitch_map.plane_homography(mats[0], poses[0], mats[1], poses[1], z)
            return cv2.warpPerspective(img, H, size)
        overhead = np.where(mask, box, table)
        front = np.where(warp(mask, 0.82) > 128, warp(box, 0.82), warp(table, 0.75))
        for role, gray in zip(stereo.roles, (overhead, front)):
            frames[role] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        async def capture(roles, settle):
            assert set(roles) == set(stereo.roles)
            return {r: frames[r].copy() for r in roles}

        monkeypatch.setattr(world, "call", lambda *a, **kw: pytest.fail("unexpected depth service call"))
        monkeypatch.setattr(segment, "segment", lambda *a, **kw: {
            "status": "ok", "results": [{"mask": mask != 0, "score": 0.9, "box": [140, 52, 185, 98]}]})
        perception = Perception(RealBody(task, frame_source=capture))
        result, _ = perception.observe(["overhead"])
        assert result["depth_source"]["source"] == "stereo_triangulation"
        assert result["depth"] == {"overhead": True}
        assert set(task.last_depth) == set(stereo.roles)
        view = perception.memory.latest("overhead")
        points = support.points_in_box(view, SafetyConfig(), {}, 0.45)
        assert support.support_height(points)["z"] == pytest.approx(0.75, abs=0.01)
        located = task.locate_object(object="box")
        assert located["status"] == "ok", located
        assert located["top_z"] == pytest.approx(0.82, abs=0.015)

        frames["front"].fill(0)
        result, _ = perception.observe(["overhead"])
        assert result["depth"] == {"overhead": False}
        assert not task.last_depth
        with pytest.raises(LookupError, match="no current depth"):
            perception._deproject("overhead", 160, 75)
        assert task.locate_object(object="box")["status"] == "no_depth"
        wrist = SimpleNamespace(frames={"right_wrist": front}, lift_m=0.45)
        assert stereo.measure(wrist, {})[0] == {}
    finally:
        loop.close()
        robot_profile.activate(None)


def test_object_top_cannot_set_table_height():
    x, y = np.mgrid[0.40:0.48:25j, 0.01:0.08:25j]
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.91)))
    with pytest.raises(support.SupportUnknown, match="wider view of bare tabletop"):
        AgentStitch.check_table(points)
