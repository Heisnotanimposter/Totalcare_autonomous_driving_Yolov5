#!/usr/bin/env python3
"""
Camera-Centric Temporal Ego-centric World Model (Stages 4-7).

Implements:
  - Explicit KITTI P2 calibration + flat-ground monocular projection with analytical
    depth uncertainty.
  - KITTI OXTS GPS/IMU Ego-motion integration (world ENU <-> Ego BEV rigid transforms)
    so Past/Present/Future object states remain geometrically consistent as Ego moves
    and turns.
  - 4D kinematic Kalman filter operating in world-stabilized coordinates (when OXTS
    is available) or ego-relative fallback, producing multi-horizon predictions
    (t + 0.5s, 1.0s, 1.5s, 2.0s) with 2x2 covariance uncertainty ellipses.
  - Uncertainty-aware Risk Indicator & advisory Safety Shield Decision where
    detection, depth, tracking, and prediction uncertainties monotonically tighten
    safety margins and escalate advisory interventions.
  - Quantitative verification against KITTI 3D GT and future observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HORIZONS_SEC: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

SAFETY_LEVELS: Tuple[str, ...] = (
    "Normal Driving",
    "Caution",
    "Speed Reduction",
    "Braking Recommendation",
    "Emergency Intervention",
)

EARTH_RADIUS_M = 6378137.0

# KITTI canonical physical heights (mean_m, std_m) for pitch-invariant monocular depth fusion
CLASS_HEIGHT_PRIORS_M: Dict[str, Tuple[float, float]] = {
    "car": (1.52, 0.16),
    "van": (2.15, 0.25),
    "truck": (3.10, 0.45),
    "pedestrian": (1.72, 0.14),
    "person_sitting": (1.28, 0.16),
    "cyclist": (1.73, 0.14),
}


def _sigmoid(x: float) -> float:
    x_clamped = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x_clamped))


@dataclass(frozen=True)
class EgoPose2D:
    """Ego pose in local ENU world coordinates (east_m, north_m, yaw_rad CC from East)."""

    frame_idx: int
    east_m: float
    north_m: float
    yaw_rad: float
    vf_mps: float = 0.0
    vl_mps: float = 0.0
    wz_radps: float = 0.0

    def ego_to_world_xy(self, x_ego: float, y_ego: float) -> Tuple[float, float]:
        """Convert Ego (X right, Y forward) -> World (East, North)."""
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        ew = self.east_m + y_ego * c + x_ego * s
        nw = self.north_m + y_ego * s - x_ego * c
        return ew, nw

    def world_to_ego_xy(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """Convert World (East, North) -> Ego (X right, Y forward)."""
        de = east_m - self.east_m
        dn = north_m - self.north_m
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        y_ego = de * c + dn * s
        x_ego = de * s - dn * c
        return x_ego, y_ego

    def ego_to_world_rot(self) -> np.ndarray:
        """2x2 orthogonal matrix R such that d[E, N]^T = R @ [X_ego, Y_ego]^T."""
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return np.array([[s, c], [-c, s]], dtype=np.float64)

    def predict_ctrv(self, dt: float) -> "EgoPose2D":
        """Predict Ego pose at t + dt using Constant Turn Rate and Velocity (CTRV)."""
        if dt <= 0:
            return self
        v = self.vf_mps
        w = self.wz_radps
        psi = self.yaw_rad
        if abs(w) < 1e-4:
            de = v * math.cos(psi) * dt
            dn = v * math.sin(psi) * dt
            new_psi = psi
        else:
            new_psi = psi + w * dt
            de = (v / w) * (math.sin(new_psi) - math.sin(psi))
            dn = (v / w) * (-math.cos(new_psi) + math.cos(psi))
        return EgoPose2D(
            frame_idx=self.frame_idx,
            east_m=self.east_m + de,
            north_m=self.north_m + dn,
            yaw_rad=new_psi,
            vf_mps=self.vf_mps,
            vl_mps=self.vl_mps,
            wz_radps=self.wz_radps,
        )


class KittiOxtsOdometry:
    """Loads per-frame Ego poses from KITTI Tracking training/oxts/XXXX.txt."""

    def __init__(self, poses_by_frame: Dict[int, EgoPose2D]):
        self.poses = poses_by_frame

    @classmethod
    def from_oxts_file(cls, path: Path) -> "KittiOxtsOdometry":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"Empty OXTS file: {path}")
        first = [float(v) for v in lines[0].split()]
        lat0, lon0 = first[0], first[1]
        scale = math.cos(math.radians(lat0))

        def mercator(lat: float, lon: float) -> Tuple[float, float]:
            mx = scale * EARTH_RADIUS_M * math.radians(lon)
            my = scale * EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
            return mx, my

        e0, n0 = mercator(lat0, lon0)
        poses: Dict[int, EgoPose2D] = {}
        for idx, ln in enumerate(lines):
            vals = [float(v) for v in ln.split()]
            ew, nw = mercator(vals[0], vals[1])
            yaw = vals[5]
            vf = vals[8] if len(vals) > 8 else 0.0
            vl = vals[9] if len(vals) > 9 else 0.0
            wz = vals[19] if len(vals) > 19 else 0.0
            poses[idx] = EgoPose2D(
                frame_idx=idx,
                east_m=ew - e0,
                north_m=nw - n0,
                yaw_rad=yaw,
                vf_mps=vf,
                vl_mps=vl,
                wz_radps=wz,
            )
        return cls(poses)

    def get_pose(self, frame_idx: int) -> Optional[EgoPose2D]:
        return self.poses.get(frame_idx)


@dataclass(frozen=True)
class KittiCalibration:
    """
    KITTI Tracking P2 camera intrinsics & mounting geometry.

    Camera coordinate frame (KITTI standard):
      X_cam: right (+), Y_cam: down (+), Z_cam: forward (+)
    Ego-centric BEV coordinate frame:
      X_ego: lateral right (+), Y_ego: longitudinal forward (+), Z_ego: height up (+)
    """

    seq_id: str
    fx: float
    fy: float
    cx: float
    cy: float
    tx: float = 0.0
    camera_height_m: float = 1.65
    pixel_noise_sigma: float = 2.0
    pitch_uncertainty_rad: float = 0.012

    @classmethod
    def from_calib_file(cls, path: Path, seq_id: str, camera_height_m: float = 1.65) -> "KittiCalibration":
        p2_vals: Optional[List[float]] = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("P2:"):
                p2_vals = [float(v) for v in line.split(":", 1)[1].split()]
                break
        if p2_vals is None or len(p2_vals) != 12:
            raise ValueError(f"Invalid or missing P2 matrix in {path}")
        return cls(
            seq_id=seq_id,
            fx=p2_vals[0],
            fy=p2_vals[5],
            cx=p2_vals[2],
            cy=p2_vals[6],
            tx=p2_vals[3],
            camera_height_m=camera_height_m,
        )

    def horizontal_fov_rad(self, image_width: int) -> float:
        return 2.0 * math.atan2(image_width / 2.0, self.fx)

    def project_bbox_ground_plane(
        self,
        bbox_xyxy: Sequence[float],
        min_z_m: float = 1.5,
        max_z_m: float = 80.0,
        class_name: Optional[str] = None,
        image_height: int = 375,
        mask_bottom_xy: Optional[Tuple[float, float]] = None,
        mask_fill_ratio: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Inverse-variance fusion of flat-ground projection and class-height prior depth.
        Uses mask bottom-contact point when an instance mask is available and falls back
        gracefully to pure ground-plane projection on vertically truncated boxes.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        use_mask_pt = (
            mask_bottom_xy is not None
            and (mask_fill_ratio is None or mask_fill_ratio >= 0.25)
            and x1 <= mask_bottom_xy[0] <= x2
            and y1 <= mask_bottom_xy[1] <= y2 + 2.0
        )
        u_mid = float(mask_bottom_xy[0]) if use_mask_pt else 0.5 * (x1 + x2)
        v_ground = float(mask_bottom_xy[1]) if use_mask_pt else y2

        dv = v_ground - self.cy
        z_ground: Optional[float] = None
        sigma_z_ground: Optional[float] = None
        if dv > 4.0:
            zg = (self.fy * self.camera_height_m) / dv
            if min_z_m <= zg <= max_z_m * 1.35:
                z_ground = zg
                sig_px = (zg * zg / (self.fy * self.camera_height_m)) * self.pixel_noise_sigma
                sig_pitch = (zg * zg / self.camera_height_m) * self.pitch_uncertainty_rad
                sigma_z_ground = max(0.45, math.hypot(sig_px, sig_pitch))

        # Pitch-invariant class-height depth prior for non-truncated boxes
        z_height: Optional[float] = None
        sigma_z_height: Optional[float] = None
        box_h_px = y2 - y1
        not_truncated_y = y1 > 4.0 and y2 < (image_height - 4.0)
        prior = CLASS_HEIGHT_PRIORS_M.get(str(class_name or "").strip().lower())
        if prior is not None and not_truncated_y and box_h_px >= 14.0 and (mask_fill_ratio is None or mask_fill_ratio >= 0.25):
            h_mean_m, h_std_m = prior
            zh = (self.fy * h_mean_m) / box_h_px
            if min_z_m <= zh <= max_z_m:
                z_height = zh
                sig_h_px = (zh * zh / (self.fy * h_mean_m)) * (1.4 * self.pixel_noise_sigma)
                sig_h_prior = zh * (h_std_m / h_mean_m)
                sigma_z_height = max(0.65, math.hypot(sig_h_px, sig_h_prior))

        if z_ground is not None and z_height is not None and sigma_z_ground is not None and sigma_z_height is not None:
            # Outlier guard: if extreme pitch shifts z_ground > 2.5*z_height, downweight z_ground
            if abs(z_ground - z_height) > 2.5 * sigma_z_height:
                sigma_z_ground *= 2.0
            wg = 1.0 / (sigma_z_ground * sigma_z_ground)
            wh = 1.0 / (sigma_z_height * sigma_z_height)
            z_m = (wg * z_ground + wh * z_height) / (wg + wh)
            sigma_z = max(0.4, math.sqrt(1.0 / (wg + wh)))
        elif z_ground is not None and sigma_z_ground is not None:
            z_m = z_ground
            sigma_z = sigma_z_ground
        elif z_height is not None and sigma_z_height is not None:
            z_m = z_height
            sigma_z = sigma_z_height
        else:
            return None

        if z_m < min_z_m or z_m > max_z_m:
            return None

        x_m = ((u_mid - self.cx) * z_m) / self.fx
        sigma_x = math.hypot((z_m / self.fx) * self.pixel_noise_sigma, ((u_mid - self.cx) / self.fx) * sigma_z)
        return x_m, z_m, max(0.25, sigma_x), max(0.4, sigma_z)


def _cov_to_ellipse_params(cov_2x2: np.ndarray) -> Tuple[float, float, float]:
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.maximum(eigvals, 1e-4)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle_deg = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return float(math.sqrt(eigvals[0])), float(math.sqrt(eigvals[1])), angle_deg


@dataclass
class ObjectWorldTrack:
    """
    Per-track temporal state & 4D kinematic Kalman filter [p1, p2, v1, v2].
    When ego-motion compensation is enabled, state lives in stabilized World ENU [E, N, vE, vN];
    otherwise it operates in Ego-relative [X_ego, Y_ego, vX, vY].
    """

    track_id: int
    class_name: str
    state: np.ndarray
    cov: np.ndarray
    first_seen_ts: float
    last_seen_ts: float
    last_seen_frame: int
    hits: int = 1
    missed_frames: int = 0
    latest_det_conf: float = 0.0
    latest_depth_sigma_m: float = 0.0
    latest_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    history: List[Dict] = field(default_factory=list)

    def predict_step(self, dt: float, accel_sigma: float = 1.5) -> None:
        if dt <= 0:
            return
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = accel_sigma * accel_sigma
        Q = q * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q

    def update(
        self,
        meas_p1: float,
        meas_p2: float,
        r_cov_2x2: np.ndarray,
        depth_sigma_m: float,
        ts: float,
        frame_idx: int,
        det_conf: float,
        class_name: str,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([meas_p1, meas_p2], dtype=np.float64)
        y_res = z - H @ self.state
        S = H @ self.cov @ H.T + r_cov_2x2
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        I = np.eye(4, dtype=np.float64)
        self.cov = (I - K @ H) @ self.cov

        self.class_name = class_name
        self.last_seen_ts = ts
        self.last_seen_frame = frame_idx
        self.hits += 1
        self.missed_frames = 0
        self.latest_det_conf = det_conf
        self.latest_depth_sigma_m = depth_sigma_m
        self.latest_bbox = bbox
        self.history.append(
            {
                "frame_idx": frame_idx,
                "timestamp_sec": round(ts, 3),
                "p1_m": float(self.state[0]),
                "p2_m": float(self.state[1]),
                "det_conf": round(det_conf, 4),
            }
        )
        if len(self.history) > 40:
            del self.history[:-40]

    def predict_future_horizons(
        self,
        current_ego_pose: Optional[EgoPose2D],
        horizons: Sequence[float] = HORIZONS_SEC,
        accel_sigma: float = 1.5,
    ) -> List[Dict]:
        """
        Propagate kinematic state & covariance over future horizons.
        Projects predictions both into current Ego BEV (for display) and CTRV future
        Ego BEV (for corridor risk & comparison with future observations/GT).
        """
        preds: List[Dict] = []
        s_curr = self.state.copy()
        p_curr = self.cov.copy()
        prev_h = 0.0
        for h in horizons:
            dt = h - prev_h
            prev_h = h
            F = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2
            q = accel_sigma * accel_sigma
            Q = q * np.array(
                [
                    [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                    [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                    [0.5 * dt3, 0.0, dt2, 0.0],
                    [0.0, 0.5 * dt3, 0.0, dt2],
                ],
                dtype=np.float64,
            )
            s_curr = F @ s_curr
            p_curr = F @ p_curr @ F.T + Q
            pos_cov = p_curr[:2, :2]

            if current_ego_pose is not None:
                ew, nw = float(s_curr[0]), float(s_curr[1])
                x_now, y_now = current_ego_pose.world_to_ego_xy(ew, nw)
                r_now = current_ego_pose.ego_to_world_rot()
                cov_ego_now = r_now.T @ pos_cov @ r_now
                sig_maj, sig_min, ang_deg = _cov_to_ellipse_params(cov_ego_now)

                fut_ego_pose = current_ego_pose.predict_ctrv(h)
                x_fut_ego, y_fut_ego = fut_ego_pose.world_to_ego_xy(ew, nw)
                r_fut = fut_ego_pose.ego_to_world_rot()
                cov_fut_ego = r_fut.T @ pos_cov @ r_fut
            else:
                ew, nw = None, None
                x_now, y_now = float(s_curr[0]), float(s_curr[1])
                x_fut_ego, y_fut_ego = x_now, y_now
                cov_ego_now = pos_cov
                cov_fut_ego = pos_cov
                sig_maj, sig_min, ang_deg = _cov_to_ellipse_params(pos_cov)

            preds.append(
                {
                    "horizon_sec": h,
                    "x_m": round(x_now, 3),
                    "y_m": round(y_now, 3),
                    "future_ego_x_m": round(x_fut_ego, 3),
                    "future_ego_y_m": round(y_fut_ego, 3),
                    "world_east_m": round(ew, 3) if ew is not None else None,
                    "world_north_m": round(nw, 3) if nw is not None else None,
                    "cov_ego_now": [[round(float(cov_ego_now[0, 0]), 4), round(float(cov_ego_now[0, 1]), 4)],
                                    [round(float(cov_ego_now[1, 0]), 4), round(float(cov_ego_now[1, 1]), 4)]],
                    "cov_future_ego": [[round(float(cov_fut_ego[0, 0]), 4), round(float(cov_fut_ego[0, 1]), 4)],
                                       [round(float(cov_fut_ego[1, 0]), 4), round(float(cov_fut_ego[1, 1]), 4)]],
                    "sigma_major_m": round(sig_maj, 3),
                    "sigma_minor_m": round(sig_min, 3),
                    "ellipse_angle_deg": round(ang_deg, 2),
                }
            )
        return preds


def compute_occlusion_sectors(
    objects: Sequence[Dict],
    max_range_m: float = 55.0,
) -> List[List[Tuple[float, float]]]:
    """Compute BEV angular shadow polygons behind directly observed objects."""
    polygons: List[List[Tuple[float, float]]] = []
    half_widths = {"Car": 0.95, "Van": 1.05, "Truck": 1.25, "Pedestrian": 0.35, "Cyclist": 0.45}
    for obj in objects:
        if obj.get("observation_state") != "Observed":
            continue
        x = float(obj["x_m"])
        y = float(obj["y_m"])
        dist = math.hypot(x, y)
        if dist < 1.5 or y <= 0.5:
            continue
        hw = half_widths.get(obj.get("class_name", "Car"), 0.8)
        theta_c = math.atan2(x, y)
        d_theta = math.atan2(hw, dist)
        t1, t2 = theta_c - d_theta, theta_c + d_theta
        r_start = dist + hw
        if r_start >= max_range_m:
            continue
        polygons.append(
            [
                (r_start * math.sin(t1), r_start * math.cos(t1)),
                (r_start * math.sin(t2), r_start * math.cos(t2)),
                (max_range_m * math.sin(t2), max_range_m * math.cos(t2)),
                (max_range_m * math.sin(t1), max_range_m * math.cos(t1)),
            ]
        )
    return polygons


class TemporalWorldModel:
    """Maintains Past -> Present -> Future Ego-centric World Model over sequential frames."""

    def __init__(
        self,
        calibration: Optional[KittiCalibration],
        oxts: Optional[KittiOxtsOdometry] = None,
        image_width: int = 1242,
        image_height: int = 375,
        max_coast_sec: float = 1.0,
        ego_corridor_half_width_m: float = 1.6,
        ego_plan_horizon_m: float = 35.0,
        uncertainty_aware: bool = True,
        uncertainty_sigma_k: float = 1.5,
        enable_risk_smoothing: bool = True,
    ):
        self.calibration = calibration
        self.oxts = oxts
        self.image_width = image_width
        self.image_height = image_height
        self.max_coast_sec = max_coast_sec
        self.ego_corridor_half_width_m = ego_corridor_half_width_m
        self.ego_plan_horizon_m = ego_plan_horizon_m
        self.uncertainty_aware = uncertainty_aware
        self.uncertainty_sigma_k = uncertainty_sigma_k
        self.enable_risk_smoothing = enable_risk_smoothing
        self.risk_smoother = CollisionRiskSmoother()
        self.tracks: Dict[int, ObjectWorldTrack] = {}
        self.last_ts: Optional[float] = None
        self.past_ego_poses: List[EgoPose2D] = []

    def step(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: Sequence[Dict],
    ) -> Dict:
        dt = (timestamp_sec - self.last_ts) if self.last_ts is not None else 0.0
        self.last_ts = timestamp_sec

        ego_pose = self.oxts.get_pose(frame_idx) if self.oxts is not None else None
        if ego_pose is not None:
            self.past_ego_poses.append(ego_pose)
            if len(self.past_ego_poses) > 40:
                del self.past_ego_poses[:-40]

        for tr in self.tracks.values():
            if dt > 0:
                tr.predict_step(dt)
            tr.missed_frames += 1

        observed_tids = set()
        uncalibrated_detections = 0

        for det in detections:
            tid = int(det["track_id"])
            if tid < 0:
                continue
            bbox = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            det_conf = float(det["conf"])
            cname = str(det["class_name"])

            if self.calibration is None:
                uncalibrated_detections += 1
                continue

            mbx = det.get("mask_bottom_x")
            mby = det.get("mask_bottom_y")
            mfr = det.get("mask_fill_ratio")
            mask_bot = (float(mbx), float(mby)) if (mbx not in (None, "") and mby not in (None, "")) else None
            mask_fill = float(mfr) if mfr not in (None, "") else None

            proj = self.calibration.project_bbox_ground_plane(
                bbox,
                class_name=cname,
                image_height=self.image_height,
                mask_bottom_xy=mask_bot,
                mask_fill_ratio=mask_fill,
            )
            if proj is None:
                continue
            mx_ego, my_ego, sx, sy = proj
            # Scale measurement uncertainty gently under low detection confidence or heavy occlusion
            conf_scale = 1.0 + max(0.0, 0.7 - det_conf) * 0.8
            if mask_fill is not None and mask_fill < 0.40:
                conf_scale *= 1.25
            sx *= conf_scale
            sy *= conf_scale
            observed_tids.add(tid)

            r_ego = np.diag([sx * sx, sy * sy])
            if ego_pose is not None:
                meas_p1, meas_p2 = ego_pose.ego_to_world_xy(mx_ego, my_ego)
                rot = ego_pose.ego_to_world_rot()
                r_cov = rot @ r_ego @ rot.T
            else:
                meas_p1, meas_p2 = mx_ego, my_ego
                r_cov = r_ego

            if tid not in self.tracks:
                init_state = np.array([meas_p1, meas_p2, 0.0, 0.0], dtype=np.float64)
                init_cov = np.zeros((4, 4), dtype=np.float64)
                init_cov[:2, :2] = r_cov
                init_cov[2:, 2:] = np.diag([9.0, 9.0])
                tr = ObjectWorldTrack(
                    track_id=tid,
                    class_name=cname,
                    state=init_state,
                    cov=init_cov,
                    first_seen_ts=timestamp_sec,
                    last_seen_ts=timestamp_sec,
                    last_seen_frame=frame_idx,
                    hits=0,
                    missed_frames=0,
                    latest_det_conf=det_conf,
                    latest_depth_sigma_m=sy,
                    latest_bbox=bbox,
                )
                self.tracks[tid] = tr

            self.tracks[tid].update(
                meas_p1, meas_p2, r_cov, sy, timestamp_sec, frame_idx, det_conf, cname, bbox
            )

        present_objects: List[Dict] = []
        expired_past: List[Dict] = []
        to_delete: List[int] = []

        for tid, tr in sorted(self.tracks.items()):
            age_since_seen = timestamp_sec - tr.last_seen_ts
            if age_since_seen > self.max_coast_sec:
                exp_hist = []
                for pt in tr.history[-15:]:
                    if ego_pose is not None:
                        hx, hy = ego_pose.world_to_ego_xy(pt["p1_m"], pt["p2_m"])
                    else:
                        hx, hy = pt["p1_m"], pt["p2_m"]
                    exp_hist.append({"x_m": round(hx, 3), "y_m": round(hy, 3), "timestamp_sec": pt["timestamp_sec"]})
                expired_past.append(
                    {
                        "track_id": tid,
                        "class_name": tr.class_name,
                        "last_seen_ts": round(tr.last_seen_ts, 3),
                        "history": exp_hist,
                    }
                )
                to_delete.append(tid)
                continue

            is_obs = tid in observed_tids
            obs_state = "Observed" if is_obs else "Occluded"
            decay = math.exp(-age_since_seen / 0.6)
            track_conf = min(1.0, (tr.hits / 5.0)) * decay
            det_unc = round(1.0 - tr.latest_det_conf, 4) if is_obs else 1.0
            track_unc = round(1.0 - track_conf, 4)

            if ego_pose is not None:
                ew, nw = float(tr.state[0]), float(tr.state[1])
                x_ego, y_ego = ego_pose.world_to_ego_xy(ew, nw)
                rot = ego_pose.ego_to_world_rot()
                pos_cov_ego = rot.T @ tr.cov[:2, :2] @ rot
                vel_cov_ego = rot.T @ tr.cov[2:, 2:] @ rot
                # World velocity expressed in current Ego axes
                v_world_in_ego = rot.T @ tr.state[2:]
                vx_world_ego, vy_world_ego = float(v_world_in_ego[0]), float(v_world_in_ego[1])
                # Relative velocity to Ego (subtract Ego forward/lateral velocity)
                vx_rel = vx_world_ego - (-ego_pose.vl_mps)
                vy_rel = vy_world_ego - ego_pose.vf_mps
            else:
                ew, nw = None, None
                x_ego, y_ego = float(tr.state[0]), float(tr.state[1])
                pos_cov_ego = tr.cov[:2, :2]
                vel_cov_ego = tr.cov[2:, 2:]
                vx_world_ego, vy_world_ego = float(tr.state[2]), float(tr.state[3])
                vx_rel, vy_rel = vx_world_ego, vy_world_ego

            depth_unc_m = round(float(math.sqrt(max(0.0, pos_cov_ego[1, 1]))), 3)
            vel_sigma_y_mps = round(float(math.sqrt(max(0.0, vel_cov_ego[1, 1]))), 3)

            calibrated_vel = self.calibration is not None and tr.hits >= 3
            speed_mps = math.hypot(vx_world_ego, vy_world_ego) if calibrated_vel else None
            rel_speed_mps = math.hypot(vx_rel, vy_rel) if calibrated_vel else None
            heading_deg = math.degrees(math.atan2(vx_world_ego, vy_world_ego)) if speed_mps is not None and speed_mps > 0.3 else None

            future_preds = tr.predict_future_horizons(ego_pose, HORIZONS_SEC) if tr.hits >= 2 else []
            pred_unc_2s = future_preds[-1]["sigma_major_m"] if future_preds else None

            past_trail = []
            for pt in tr.history:
                dt_hist = max(0.0, timestamp_sec - pt["timestamp_sec"])
                if ego_pose is not None:
                    hx, hy = ego_pose.world_to_ego_xy(pt["p1_m"], pt["p2_m"])
                else:
                    hx, hy = pt["p1_m"], pt["p2_m"]
                past_trail.append(
                    {
                        "frame_idx": pt["frame_idx"],
                        "timestamp_sec": pt["timestamp_sec"],
                        "x_m": round(hx, 3),
                        "y_m": round(hy, 3),
                        "temporal_weight": round(math.exp(-dt_hist / 1.5), 3),
                    }
                )

            present_objects.append(
                {
                    "track_id": tid,
                    "class_name": tr.class_name,
                    "observation_state": obs_state,
                    "x_m": round(x_ego, 3),
                    "y_m": round(y_ego, 3),
                    "world_east_m": round(ew, 3) if ew is not None else None,
                    "world_north_m": round(nw, 3) if nw is not None else None,
                    "vx_mps": round(vx_world_ego, 3) if calibrated_vel else None,
                    "vy_mps": round(vy_world_ego, 3) if calibrated_vel else None,
                    "vx_rel_mps": round(vx_rel, 3) if calibrated_vel else None,
                    "vy_rel_mps": round(vy_rel, 3) if calibrated_vel else None,
                    "speed_mps": round(speed_mps, 3) if speed_mps is not None else None,
                    "rel_speed_mps": round(rel_speed_mps, 3) if rel_speed_mps is not None else None,
                    "heading_deg": round(heading_deg, 1) if heading_deg is not None else None,
                    "last_seen_ts": round(tr.last_seen_ts, 3),
                    "hits": tr.hits,
                    "missed_frames": tr.missed_frames,
                    "uncertainties": {
                        "detection_uncertainty": det_unc,
                        "depth_uncertainty_m": depth_unc_m,
                        "tracking_uncertainty": track_unc,
                        "velocity_uncertainty_y_mps": vel_sigma_y_mps,
                        "motion_prediction_uncertainty_2s_m": pred_unc_2s,
                    },
                    "past_trail": past_trail,
                    "future_predictions": future_preds,
                }
            )

        for tid in to_delete:
            del self.tracks[tid]

        # Ego CTRV planned path expressed in current Ego coordinates
        ego_planned_path: List[Dict] = []
        for h in (0.0, *HORIZONS_SEC):
            if ego_pose is not None:
                fut_p = ego_pose.predict_ctrv(h)
                px, py = ego_pose.world_to_ego_xy(fut_p.east_m, fut_p.north_m)
                ego_planned_path.append({"horizon_sec": h, "x_m": round(px, 2), "y_m": round(py, 2)})
            else:
                ego_planned_path.append({"horizon_sec": h, "x_m": 0.0, "y_m": round(min(self.ego_plan_horizon_m, 10.0 * h), 2)})

        # Ego driven past path expressed in current Ego coordinates
        ego_past_trail: List[Dict] = []
        if ego_pose is not None:
            for pp in self.past_ego_poses:
                px, py = ego_pose.world_to_ego_xy(pp.east_m, pp.north_m)
                ego_past_trail.append({"x_m": round(px, 3), "y_m": round(py, 3)})

        occlusion_sectors = compute_occlusion_sectors(present_objects)
        risk_assessment = assess_risk_and_safety_shield(
            present_objects=present_objects,
            calibrated=self.calibration is not None,
            corridor_half_width_m=self.ego_corridor_half_width_m,
            uncertainty_aware=self.uncertainty_aware,
            uncertainty_sigma_k=self.uncertainty_sigma_k,
        )
        if self.enable_risk_smoothing and self.calibration is not None:
            risk_assessment = self.risk_smoother.smooth(risk_assessment, dt=dt)

        fov_deg = round(math.degrees(self.calibration.horizontal_fov_rad(self.image_width)), 1) if self.calibration else None

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "calibrated": self.calibration is not None,
            "ego_motion_compensated": ego_pose is not None,
            "uncalibrated_detections_skipped": uncalibrated_detections,
            "coordinate_system": "ego_centric_ground_plane_x_right_y_forward",
            "fov_deg": fov_deg,
            "ego": {
                "x_m": 0.0,
                "y_m": 0.0,
                "world_east_m": round(ego_pose.east_m, 3) if ego_pose else None,
                "world_north_m": round(ego_pose.north_m, 3) if ego_pose else None,
                "yaw_deg": round(math.degrees(ego_pose.yaw_rad), 2) if ego_pose else None,
                "speed_mps": round(ego_pose.vf_mps, 2) if ego_pose else None,
                "yaw_rate_degps": round(math.degrees(ego_pose.wz_radps), 2) if ego_pose else None,
                "corridor_half_width_m": self.ego_corridor_half_width_m,
                "planned_path": ego_planned_path,
                "past_trail": ego_past_trail,
            },
            "past_world": {
                "expired_tracks": expired_past,
            },
            "present_world": {
                "objects": present_objects,
                "occlusion_sectors": occlusion_sectors,
            },
            "future_world": {
                "horizons_sec": list(HORIZONS_SEC),
                "ego_planned_path": ego_planned_path,
                "conflict_zones": risk_assessment["conflict_zones"],
            },
            "risk_assessment": risk_assessment,
        }


def _risk_to_decision(risk_val: float, min_ttc: Optional[float]) -> str:
    if risk_val >= 0.85 or (min_ttc is not None and min_ttc <= 1.2):
        return "Emergency Intervention"
    if risk_val >= 0.65 or (min_ttc is not None and min_ttc <= 2.0):
        return "Braking Recommendation"
    if risk_val >= 0.45:
        return "Speed Reduction"
    if risk_val >= 0.25:
        return "Caution"
    return "Normal Driving"


class CollisionRiskSmoother:
    """
    Temporal gradient-bounded smoother for spiking collision/risk detectors.
    Combines asymmetric EMA filtering, explicit |dR/dt| slew-rate clamping, and
    hysteresis debouncing on advisory Safety Shield transitions.
    """

    def __init__(
        self,
        alpha_up: float = 0.42,
        alpha_down: float = 0.22,
        max_rise_per_sec: float = 1.25,
        max_fall_per_sec: float = 0.85,
        hysteresis_margin: float = 0.03,
    ):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_rise_per_sec = max_rise_per_sec
        self.max_fall_per_sec = max_fall_per_sec
        self.hysteresis_margin = hysteresis_margin
        self.prev_raw: Optional[float] = None
        self.prev_smooth: Optional[float] = None
        self.prev_decision: str = "Normal Driving"
        self.per_track_ema: Dict[int, float] = {}

    def smooth(self, raw_assessment: Dict, dt: float) -> Dict:
        raw_risk = float(raw_assessment["max_risk_indicator"])
        min_ttc = raw_assessment.get("min_calibrated_ttc_sec")

        if self.prev_smooth is None or dt <= 0.0:
            smoothed = raw_risk
            raw_grad = 0.0
            smooth_grad = 0.0
        else:
            alpha = self.alpha_up if raw_risk >= self.prev_smooth else self.alpha_down
            ema_val = alpha * raw_risk + (1.0 - alpha) * self.prev_smooth
            max_up = self.max_rise_per_sec * dt
            max_dn = self.max_fall_per_sec * dt
            delta = max(-max_dn, min(max_up, ema_val - self.prev_smooth))
            smoothed = max(0.0, min(1.0, self.prev_smooth + delta))
            raw_grad = (raw_risk - (self.prev_raw if self.prev_raw is not None else raw_risk)) / dt
            smooth_grad = (smoothed - self.prev_smooth) / dt

        self.prev_raw = raw_risk
        self.prev_smooth = smoothed

        # Hysteresis debouncing on decision downgrades to avoid threshold chattering
        rank_map = {lvl: idx for idx, lvl in enumerate(SAFETY_LEVELS)}
        thresholds = [0.0, 0.25, 0.45, 0.65, 0.85]
        cand = _risk_to_decision(smoothed, min_ttc)
        prev_rank = rank_map[self.prev_decision]
        cand_rank = rank_map[cand]
        if cand_rank < prev_rank:
            # Downgrade only if smoothed risk stays comfortably below previous threshold - hysteresis_margin
            stay_thresh = thresholds[prev_rank] - self.hysteresis_margin
            if smoothed >= stay_thresh:
                cand = self.prev_decision
        self.prev_decision = cand

        out = dict(raw_assessment)
        out["raw_max_risk_indicator"] = round(raw_risk, 3)
        out["max_risk_indicator"] = round(smoothed, 3)
        out["safety_shield_decision"] = cand
        out["risk_gradient_raw_per_sec"] = round(raw_grad, 3)
        out["risk_gradient_smoothed_per_sec"] = round(smooth_grad, 3)
        out["risk_smoothing_enabled"] = True
        return out


def assess_risk_and_safety_shield(
    present_objects: List[Dict],
    calibrated: bool,
    corridor_half_width_m: float = 1.6,
    uncertainty_aware: bool = True,
    uncertainty_sigma_k: float = 1.5,
) -> Dict:
    """
    Compute smooth continuous Risk Indicator (NOT collision probability) and advisory
    Safety Shield Decision using differentiable sigmoid/soft-union hazard potentials.
    """
    if not calibrated:
        return {
            "raw_max_risk_indicator": 0.0,
            "max_risk_indicator": 0.0,
            "min_calibrated_ttc_sec": None,
            "safety_shield_decision": "Normal Driving",
            "decision_scope": "advisory_only_uncalibrated_no_ttc",
            "uncertainty_aware": uncertainty_aware,
            "risk_smoothing_enabled": False,
            "conflict_zones": [],
        }

    k_sig = uncertainty_sigma_k if uncertainty_aware else 0.0
    max_risk = 0.0
    min_ttc: Optional[float] = None
    conflict_zones: List[Dict] = []

    for obj in present_objects:
        x = float(obj["x_m"])
        y = float(obj["y_m"])
        unc = obj["uncertainties"]
        depth_sig = float(unc["depth_uncertainty_m"])
        det_unc = float(unc["detection_uncertainty"])
        track_unc = float(unc["tracking_uncertainty"])
        vel_sig_y = float(unc.get("velocity_uncertainty_y_mps", 0.0))
        pred_unc_2s = float(unc["motion_prediction_uncertainty_2s_m"] or depth_sig)

        vy_rel = obj.get("vy_rel_mps") if obj.get("vy_rel_mps") is not None else obj.get("vy_mps")

        # Smooth lateral corridor membership weight C1-continuous across boundary
        lat_envelope = corridor_half_width_m + min(2.0, k_sig * 0.5 * depth_sig) + 0.55
        w_lat = _sigmoid((lat_envelope - abs(x)) / 0.45) if y > 0.0 else 0.0
        in_corridor_now = y > 0.0 and (abs(x) - (corridor_half_width_m + min(2.0, k_sig * 0.5 * depth_sig))) <= 0.6

        # Strict calibrated TTC for reporting (only when inside corridor & closing)
        obj_ttc: Optional[float] = None
        y_cons = max(0.2, y - min(4.0, k_sig * 0.5 * depth_sig))
        if in_corridor_now and vy_rel is not None and vy_rel < -0.5 and y > 0.5:
            closing_cons = abs(vy_rel) + k_sig * 0.5 * vel_sig_y
            obj_ttc = round(y_cons / max(0.5, closing_cons), 2)
            if min_ttc is None or obj_ttc < min_ttc:
                min_ttc = obj_ttc

        # Smooth TTC hazard potential (avoids step discontinuity around vy_rel = -0.5 m/s)
        ttc_score = 0.0
        if vy_rel is not None and y > 0.3:
            w_close = _sigmoid(((-vy_rel) - 0.35) / 0.35)
            closing_soft = max(0.25, max(0.0, -vy_rel) + k_sig * 0.45 * vel_sig_y)
            ttc_soft = y_cons / closing_soft
            ttc_score = w_lat * w_close * _sigmoid((3.4 - ttc_soft) / 0.70)

        # Future trajectory intersection check in CTRV-predicted future Ego corridor
        min_pred_corridor_dist = abs(x) if y > 0 else 99.0
        best_conflict_pt: Optional[Dict] = None
        for fp in obj.get("future_predictions", []):
            fx_fut = float(fp.get("future_ego_x_m", fp["x_m"]))
            fy_fut = float(fp.get("future_ego_y_m", fp["y_m"]))
            fx_now = float(fp["x_m"])
            fy_now = float(fp["y_m"])
            fsig = float(fp["sigma_major_m"])
            if 0.0 < fy_fut < 35.0:
                eff_lat = max(0.0, abs(fx_fut) - corridor_half_width_m - min(2.5, k_sig * 0.5 * fsig))
                if eff_lat < min_pred_corridor_dist:
                    min_pred_corridor_dist = eff_lat
                if eff_lat <= 0.5:
                    best_conflict_pt = {
                        "track_id": obj["track_id"],
                        "horizon_sec": fp["horizon_sec"],
                        "x_m": fx_now,
                        "y_m": fy_now,
                        "radius_m": round(max(1.0, fsig), 2),
                    }

        y_eff = max(0.5, y - min(4.0, k_sig * 0.5 * depth_sig))
        dist_score = w_lat * _sigmoid((20.0 - y_eff) / 5.5) if (0.0 < y < 38.0) else 0.0
        pred_score = _sigmoid((2.0 - min_pred_corridor_dist) / 0.85) * _sigmoid((36.0 - y) / 4.0) if (0.0 < y < 42.0) else 0.0
        base_risk = 1.0 - (1.0 - 0.78 * dist_score) * (1.0 - 0.65 * pred_score) * (1.0 - 0.94 * ttc_score)

        # Epistemic & perception uncertainty risk inflation for forward near-corridor objects
        epistemic_boost = 0.0
        if uncertainty_aware and 0.0 < y < 38.0 and abs(x) < (corridor_half_width_m + 3.5):
            epistemic_unc = 0.45 * det_unc + 0.35 * track_unc + 0.20 * min(1.0, pred_unc_2s / 6.0)
            prox_w = _sigmoid((24.0 - y) / 6.0)
            lat_w = max(0.25, 1.0 - abs(x) / (corridor_half_width_m + 3.5))
            epistemic_boost = 0.40 * epistemic_unc * prox_w * lat_w

        obj_risk = round(min(1.0, base_risk + epistemic_boost), 3)
        obj["risk_indicator"] = obj_risk
        obj["calibrated_ttc_sec"] = obj_ttc
        obj["epistemic_risk_boost"] = round(epistemic_boost, 3)

        if obj_risk > max_risk:
            max_risk = obj_risk
        if best_conflict_pt is not None and obj_risk >= 0.35:
            conflict_zones.append(best_conflict_pt)

    decision = _risk_to_decision(max_risk, min_ttc)

    return {
        "raw_max_risk_indicator": round(max_risk, 3),
        "max_risk_indicator": round(max_risk, 3),
        "min_calibrated_ttc_sec": min_ttc,
        "safety_shield_decision": decision,
        "decision_scope": "advisory_safety_shield_not_physical_actuation",
        "uncertainty_aware": uncertainty_aware,
        "risk_smoothing_enabled": False,
        "conflict_zones": conflict_zones,
    }


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def evaluate_world_model_against_kitti_gt(
    world_frames: Sequence[Dict],
    detections_by_frame: Dict[int, List[Dict]],
    label_txt_lines: Iterable[str],
    oxts: Optional[KittiOxtsOdometry] = None,
    fps: float = 10.0,
) -> Dict:
    """
    Evaluate monocular depth (Abs Rel, RMSE), multi-horizon prediction (ADE, FDE)
    in both Future-Ego frame and World frame, and 1-sigma / 2-sigma Gaussian ellipse
    calibration coverage against KITTI label_02 3D ground truth.
    """
    gt_by_frame: Dict[int, List[Dict]] = {}
    gt_by_track_frame: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for raw in label_txt_lines:
        parts = raw.strip().split()
        if len(parts) < 17:
            continue
        fr = int(parts[0])
        gt_tid = int(parts[1])
        cname = parts[2]
        if cname == "DontCare" or gt_tid < 0:
            continue
        bbox = (float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]))
        gx_ego = float(parts[13])
        gz_ego = float(parts[15])
        if gz_ego <= 0.5:
            continue
        gt_by_frame.setdefault(fr, []).append(
            {"gt_tid": gt_tid, "class_name": cname, "bbox": bbox, "x_m": gx_ego, "y_m": gz_ego}
        )
        gt_by_track_frame[(gt_tid, fr)] = (gx_ego, gz_ego)

    abs_rel_errs: List[float] = []
    sq_errs: List[float] = []
    ade_ego_errs: List[float] = []
    fde_ego_errs: List[float] = []
    ade_world_errs: List[float] = []
    fde_world_errs: List[float] = []
    per_horizon_ego: Dict[float, List[float]] = {h: [] for h in HORIZONS_SEC}
    per_horizon_world: Dict[float, List[float]] = {h: [] for h in HORIZONS_SEC}
    inside_1sig = 0
    inside_2sig = 0
    cov_samples = 0
    matched_depth_samples = 0

    for wf in world_frames:
        fr = int(wf["frame_idx"])
        gt_objs = gt_by_frame.get(fr, [])
        if not gt_objs:
            continue
        det_map = {int(d["track_id"]): d for d in detections_by_frame.get(fr, []) if int(d["track_id"]) >= 0}

        for obj in wf.get("present_world", {}).get("objects", []):
            if obj.get("observation_state") != "Observed":
                continue
            tid = int(obj["track_id"])
            det = det_map.get(tid)
            if det is None:
                continue
            det_box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))

            best_gt = None
            best_iou = 0.4
            for g in gt_objs:
                iou = _iou_xyxy(det_box, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g
            if best_gt is None:
                continue

            pred_z = float(obj["y_m"])
            gt_z = float(best_gt["y_m"])
            abs_rel_errs.append(abs(pred_z - gt_z) / max(0.5, gt_z))
            sq_errs.append((pred_z - gt_z) ** 2)
            matched_depth_samples += 1

            preds = obj.get("future_predictions", [])
            if preds:
                step_ego = []
                step_world = []
                for fp in preds:
                    h = float(fp["horizon_sec"])
                    f_offset = int(round(h * fps))
                    gt_fut = gt_by_track_frame.get((best_gt["gt_tid"], fr + f_offset))
                    if gt_fut is None:
                        continue

                    fx_e = float(fp.get("future_ego_x_m", fp["x_m"]))
                    fy_e = float(fp.get("future_ego_y_m", fp["y_m"]))
                    err_e = math.hypot(fx_e - gt_fut[0], fy_e - gt_fut[1])
                    step_ego.append(err_e)
                    if h in per_horizon_ego:
                        per_horizon_ego[h].append(err_e)

                    # Evaluate in World ENU if OXTS is available (decouples object motion from future Ego turns)
                    if oxts is not None and fp.get("world_east_m") is not None:
                        gt_pose_fut = oxts.get_pose(fr + f_offset)
                        if gt_pose_fut is not None:
                            gt_ew, gt_nw = gt_pose_fut.ego_to_world_xy(gt_fut[0], gt_fut[1])
                            err_w = math.hypot(float(fp["world_east_m"]) - gt_ew, float(fp["world_north_m"]) - gt_nw)
                            step_world.append(err_w)
                            if h in per_horizon_world:
                                per_horizon_world[h].append(err_w)

                    # Mahalanobis uncertainty calibration check in Future-Ego frame
                    cov_mat = np.array(fp.get("cov_future_ego", fp.get("cov_ego_now")), dtype=np.float64)
                    diff = np.array([gt_fut[0] - fx_e, gt_fut[1] - fy_e], dtype=np.float64)
                    try:
                        maha2 = float(diff.T @ np.linalg.inv(cov_mat) @ diff)
                        # 2D chi-square thresholds: 1-sigma (39.3%) = 2.279, 2-sigma (86.5%) = 5.991
                        if maha2 <= 2.279:
                            inside_1sig += 1
                        if maha2 <= 5.991:
                            inside_2sig += 1
                        cov_samples += 1
                    except np.linalg.LinAlgError:
                        pass

                if step_ego:
                    ade_ego_errs.append(sum(step_ego) / len(step_ego))
                    fde_ego_errs.append(step_ego[-1])
                if step_world:
                    ade_world_errs.append(sum(step_world) / len(step_world))
                    fde_world_errs.append(step_world[-1])

    return {
        "depth_metrics": {
            "matched_samples": matched_depth_samples,
            "abs_rel": round(float(np.mean(abs_rel_errs)), 4) if abs_rel_errs else None,
            "rmse_m": round(float(math.sqrt(np.mean(sq_errs))), 4) if sq_errs else None,
        },
        "motion_prediction_metrics": {
            "matched_trajectories": len(ade_ego_errs),
            "ade_future_ego_m": round(float(np.mean(ade_ego_errs)), 4) if ade_ego_errs else None,
            "fde_future_ego_m": round(float(np.mean(fde_ego_errs)), 4) if fde_ego_errs else None,
            "ade_world_m": round(float(np.mean(ade_world_errs)), 4) if ade_world_errs else None,
            "fde_world_m": round(float(np.mean(fde_world_errs)), 4) if fde_world_errs else None,
            "per_horizon_future_ego_m": {
                str(h): round(float(np.mean(vals)), 4) if vals else None for h, vals in per_horizon_ego.items()
            },
            "per_horizon_world_m": {
                str(h): round(float(np.mean(vals)), 4) if vals else None for h, vals in per_horizon_world.items()
            },
        },
        "uncertainty_calibration": {
            "samples": cov_samples,
            "empirical_1sigma_coverage": round(inside_1sig / cov_samples, 4) if cov_samples else None,
            "empirical_2sigma_coverage": round(inside_2sig / cov_samples, 4) if cov_samples else None,
            "theoretical_2d_gaussian": {"1sigma": 0.3935, "2sigma": 0.8647},
        },
        "note": "Evaluated against KITTI label_02 3D GT matched via IoU > 0.4; World-frame uses OXTS Ego poses.",
    }
