"""Depth from corresponding RGB patches in the calibrated overhead and front cameras."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from perception.rough_plane import BASE_LINK_ABOVE_FOOTPRINT
from robot.stereo_cursor import StereoCursor
from synthesis.real import stitch_map, support


class AgentStitch:
    roles = stitch_map.ROLES

    def __init__(self, robot):
        folder = Path(stitch_map.__file__).resolve().parents[2] / "calibration" / robot.lower()
        self.cursor = StereoCursor.from_files(folder / "calibration.json", folder / "triangulation_result.json")
        # OpenCV's stereo solve maps overhead optical coordinates into front coordinates.
        front_from_overhead = np.eye(4)
        front_from_overhead[:3, :3] = self.cursor.calib["R"]
        front_from_overhead[:3, 3] = np.ravel(self.cursor.calib["T"])
        self.overhead_from_front = np.linalg.inv(front_from_overhead)
        # 0003, 2026-11-21: fit front rotation on the 0.74 m tabletop with camera centres fixed.
        if robot.upper() == "NORI-A3-0003":
            self.overhead_from_front[:3, :3] = self.overhead_from_front[:3, :3] @ stitch_map._rot((0, 3.38, 0))

    def pose(self, role, lift):
        transform = np.eye(4)
        rotation, translation = self.cursor.pose_at_lift(lift)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        # StereoCursor is calibrated in base_link. Agent targets and the guard use base_footprint.
        transform[2, 3] += BASE_LINK_ABOVE_FOOTPRINT
        return transform if role == "overhead" else transform @ self.overhead_from_front

    def intrinsics(self, role, size=None):
        size = size or self.cursor.image_size
        camera = self.cursor.for_frame_size(size)
        matrix = np.asarray(camera.calib["K_a" if role == "overhead" else "K_b"])
        return (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]), size

    def rectify(self, obs, intrinsics):
        for role in self.roles:
            if role not in obs.frames:
                continue
            frame = obs.frames[role]
            height, width = frame.shape[:2]
            intrinsics[role], _ = self.intrinsics(role, (width, height))
            matrix = stitch_map._Kmat(intrinsics[role])
            distortion = self.cursor.calib["dist_a" if role == "overhead" else "dist_b"]
            obs.frames[role] = cv2.undistort(frame, matrix, np.asarray(distortion))

    def attach(self, task, safety_config=None):
        """Use the task's local stereo hook for both observation entry points."""
        task.stitch = self
        task.use_service = False
        task.depth_model = "stereo_triangulation"
        task.table_anchor = False

        def depth(obs):
            task.obs_base = list(getattr(task.robot, "base", (0, 0, 0)))[:3]
            maps, task.last_stereo_diag = self.measure(obs, task._K)
            if safety_config is not None and safety_config.table_z is None and "overhead" in maps:
                view = SimpleNamespace(depth=maps["overhead"]["depth_m"], conf=None, K=task._K["overhead"],
                                       T_base_cam=self.pose("overhead", obs.lift_m))
                points = support.points_in_box(view, safety_config, obs.q, obs.lift_m)
                self.check_table(points)
            return maps

        task._stereo_depth = depth

    @staticmethod
    def check_table(points):
        """A single box top can pass the policy worker's flat-layer check."""
        surface = support.support_height(points)
        if surface["kind"] != "table":
            raise support.SupportUnknown("the stereo pair sees the floor but could not establish the tabletop")
        xy = points[np.abs(points[:, 2] - surface["z"]) <= 0.03, :2]
        centered = xy - xy.mean(axis=0)
        axes = np.linalg.svd(centered, full_matrices=False)[2]
        spans = np.ptp(np.percentile(centered @ axes.T, [5, 95], axis=0), axis=0)
        if max(spans) < 0.20 or min(spans) < 0.12:
            raise support.SupportUnknown(f"flat surface at {surface['z']:.3f} m spans only "
                                         f"{spans[0]:.2f} × {spans[1]:.2f} m in the hand workspace; "
                                         "both head cameras need a wider view of bare tabletop")

    def measure(self, obs, intrinsics):
        diag = {"source": "stereo_triangulation", "method": "front_overhead_stitch", "accepted": False}
        if obs.lift_m is None or not all(r in obs.frames for r in self.roles):
            return {}, dict(diag, why="depth needs overhead and front RGB from the same observation and the lift position")

        # Register the two views in image space, then triangulate the corresponding
        # original pixels. The registration supplies no metric scale or scene height.
        poses = [self.pose(r, obs.lift_m) for r in self.roles]
        projection = [stitch_map._Kmat(intrinsics[r]) @ np.linalg.inv(pose)[:3]
                      for r, pose in zip(self.roles, poses)]
        images = [cv2.cvtColor(obs.frames[r], cv2.COLOR_RGB2GRAY) for r in self.roles]
        origin, target = _correspondences(*images)
        if len(origin) < stitch_map.MIN_POINTS:
            return {}, dict(diag, why="too few consistent two-view correspondences", matches=len(origin))
        homogeneous = cv2.triangulatePoints(*projection, origin.T, target.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            points = homogeneous / homogeneous[3]
            rays = [P @ points for P in projection]
            good = np.all(np.isfinite(points), axis=0)
            sightlines = [points[:3].T - pose[:3, 3] for pose in poses]
            directions = [line / np.linalg.norm(line, axis=1, keepdims=True) for line in sightlines]
            good &= np.sum(directions[0] * directions[1], axis=1) < np.cos(np.deg2rad(stitch_map.MIN_PARALLAX_DEG))
            for ray, measured in zip(rays, (origin, target)):
                lo, hi = stitch_map.DEPTH_RANGE_M
                good &= (ray[2] >= lo) & (ray[2] <= hi)
                good &= np.linalg.norm((ray[:2] / ray[2]).T - measured, axis=1) <= stitch_map.REPROJ_PX
        if good.sum() < stitch_map.MIN_POINTS:
            return {}, dict(diag, why="too few triangulated pixels", matches=int(good.sum()))
        maps = {}
        for role, ray, pixels, img in zip(self.roles, rays, (origin, target), images):
            depth = np.full(img.shape, np.inf, np.float32)
            cols, rows = np.rint(pixels[good]).astype(int).T
            np.minimum.at(depth, (rows, cols), ray[2, good])
            depth[~np.isfinite(depth)] = 0
            maps[role] = {"depth_m": depth}
        return maps, dict(diag, accepted=True, matches=int(good.sum()))


def _correspondences(overhead, front):
    """Register the dominant shared surface, then match textured patches with bidirectional flow."""
    empty = np.empty((0, 2), np.float32)
    a, b = _features(overhead, front)
    if len(a) < 8:
        return empty, empty
    H, inliers = cv2.findHomography(b, a, cv2.RANSAC, 2.5)
    if H is None or inliers.sum() < 8 or np.linalg.cond(H) > 1e8:
        return empty, empty
    first, second = _flow_matches(overhead, front, H)
    # Once the main viewpoint difference is removed, features on a raised object
    # can be matched separately. A tabletop warp alone can miss the entire box.
    aligned = cv2.warpPerspective(front, H, overhead.shape[::-1])
    a, b = _features(overhead, aligned)
    displaced = np.linalg.norm(a - b, axis=1) > 3
    if displaced.sum() >= 8:
        correction, inliers = cv2.findHomography(b[displaced], a[displaced], cv2.RANSAC, 2)
        if correction is not None and inliers.sum() >= 8 and np.linalg.cond(correction) < 1e8:
            more, corresponding = _flow_matches(overhead, front, correction @ H)
            # Refuse conflicting correspondences at a pixel across the two registrations.
            points = np.full((*overhead.shape, 2), np.nan, np.float32)
            x, y = first.astype(int).T
            points[y, x] = second
            x, y = more.astype(int).T
            conflict = np.linalg.norm(points[y, x] - corresponding, axis=1) > 2
            points[y, x] = corresponding
            points[y[conflict], x[conflict]] = np.nan
            y, x = np.nonzero(np.isfinite(points[:, :, 0]))
            return np.column_stack((x, y)).astype(np.float32), points[y, x]
    return first, second


def _features(left, right):
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.012)
    ka, da = sift.detectAndCompute(left, None)
    kb, db = sift.detectAndCompute(right, None)
    if da is None or db is None or len(db) < 2:
        return (np.empty((0, 2), np.float32),) * 2
    matcher = cv2.BFMatcher()
    back = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    pairs = [m for m, n in matcher.knnMatch(da, db, k=2)
             if m.distance < 0.85 * n.distance and back[m.trainIdx] == m.queryIdx]
    return (np.float32([ka[m.queryIdx].pt for m in pairs]).reshape(-1, 2),
            np.float32([kb[m.trainIdx].pt for m in pairs]).reshape(-1, 2))


def _flow_matches(overhead, front, H):
    empty = np.empty((0, 2), np.float32)
    h, w = overhead.shape
    warped = cv2.warpPerspective(front, H, (w, h))
    mask = cv2.warpPerspective(np.full_like(front, 255), H, (w, h))
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8))
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(0)
    fwd = flow.calc(overhead, warped, None)
    rev = flow.calc(warped, overhead, None)
    grid = np.indices((h, w), np.float32)[::-1].transpose(1, 2, 0)
    locations = grid + fwd
    reverse = cv2.remap(rev, locations, None, cv2.INTER_LINEAR)
    valid = np.linalg.norm(fwd + reverse, axis=2) <= 0.8
    valid &= cv2.remap(mask, locations, None, cv2.INTER_NEAREST) == 255
    # Flow interpolates through flat areas. Keep only pixels with visible texture
    # and a photometrically consistent patch in the actual second image.
    aligned = cv2.remap(warped, locations, None, cv2.INTER_LINEAR).astype(np.float32)
    ref = overhead.astype(np.float32)
    def mean(img):
        return cv2.blur(img, (7, 7))
    ma, mb = mean(ref), mean(aligned)
    va = np.maximum(0, mean(ref * ref) - ma * ma)
    vb = np.maximum(0, mean(aligned * aligned) - mb * mb)
    valid &= (va > 25) & (vb > 25)
    valid &= mean(ref * aligned) - ma * mb > 0.90 * np.sqrt(va * vb)
    valid &= cv2.cornerMinEigenVal(overhead, 5) > 0.00012
    if not valid.any():
        return empty, empty
    original = cv2.perspectiveTransform(locations[valid, None], np.linalg.inv(H))[:, 0]
    h, w = front.shape
    inside = (original[:, 0] >= 5) & (original[:, 0] < w - 5) & (original[:, 1] >= 5) & (original[:, 1] < h - 5)
    return grid[valid][inside], original[inside]
