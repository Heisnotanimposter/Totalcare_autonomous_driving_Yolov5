#!/usr/bin/env python3
"""
Scientific Validation Suite for Camera-Centric Temporal Ego-centric World Model.

Answers three foundational research questions quantitatively on KITTI Tracking:
  1. Coordinate & Ego-Motion Consistency:
     - Exact round-trip world <-> ego transform error
     - Stationary-object stability (world-compensated vs raw ego-relative drift)
  2. Future Prediction vs Real Future Observations:
     - 3-way ablation (Zero-Velocity vs Ego-Relative Kalman vs OXTS Ego-Compensated World Kalman)
     - Per-horizon (+0.5s, +1.0s, +1.5s, +2.0s) ADE/FDE against KITTI 3D GT + self-consistency
     - Empirical 1-sigma / 2-sigma uncertainty ellipse coverage calibration
  3. Uncertainty Causality on Risk Assessment & Safety Shield:
     - Counterfactual A/B/C comparison (Uncertainty-Blind vs Uncertainty-Aware Nominal vs
       Degraded Perception with low confidence, bbox jitter & occlusion dropouts)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_model import (
    HORIZONS_SEC,
    SAFETY_LEVELS,
    KittiCalibration,
    KittiOxtsOdometry,
    TemporalWorldModel,
    evaluate_world_model_against_kitti_gt,
)

SAFETY_RANK: Dict[str, int] = {lvl: idx for idx, lvl in enumerate(SAFETY_LEVELS)}


def load_trajectories_csv(path: Path) -> Dict[int, List[Dict]]:
    dets_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sfr = int(row.get("source_frame_idx", row["frame_idx"]))
            row["track_id"] = int(row["track_id"])
            for k in ("conf", "x1", "y1", "x2", "y2"):
                row[k] = float(row[k])
            for mk in ("mask_area_px", "mask_fill_ratio", "mask_bottom_x", "mask_bottom_y"):
                val = row.get(mk, "")
                if val is not None and str(val).strip() != "":
                    row[mk] = float(val)
                else:
                    row[mk] = None
            dets_by_frame[sfr].append(row)
    return dict(dets_by_frame)


def run_world_model_pass(
    frames: Sequence[int],
    dets_by_frame: Dict[int, List[Dict]],
    calib: KittiCalibration,
    oxts: Optional[KittiOxtsOdometry],
    fps: float = 10.0,
    image_width: int = 1242,
    image_height: int = 375,
    uncertainty_aware: bool = True,
    enable_risk_smoothing: bool = True,
) -> List[Dict]:
    wm = TemporalWorldModel(
        calibration=calib,
        oxts=oxts,
        image_width=image_width,
        image_height=image_height,
        uncertainty_aware=uncertainty_aware,
        enable_risk_smoothing=enable_risk_smoothing,
    )
    timeline: List[Dict] = []
    base_fr = frames[0]
    for fr in frames:
        ts = (fr - base_fr) / fps
        timeline.append(wm.step(frame_idx=fr, timestamp_sec=ts, detections=dets_by_frame.get(fr, [])))
    return timeline


def verify_coordinate_transforms_and_ego_compensation(
    frames: Sequence[int],
    dets_by_frame: Dict[int, List[Dict]],
    calib: KittiCalibration,
    oxts: KittiOxtsOdometry,
    label_lines: Sequence[str],
    fps: float = 10.0,
) -> Dict:
    """
    RQ1: Verify coordinate transform accuracy & stationary object stability under Ego motion.
    """
    # 1a. Algebraic round-trip transform precision across all frames
    max_rt_err = 0.0
    ego_path_len = 0.0
    prev_pose = None
    for fr in frames:
        pose = oxts.get_pose(fr)
        if pose is None:
            continue
        for tx, ty in [(0.0, 10.0), (-5.5, 22.3), (8.2, 41.0)]:
            ew, nw = pose.ego_to_world_xy(tx, ty)
            rx, ry = pose.world_to_ego_xy(ew, nw)
            max_rt_err = max(max_rt_err, math.hypot(rx - tx, ry - ty))
        if prev_pose is not None:
            ego_path_len += math.hypot(pose.east_m - prev_pose.east_m, pose.north_m - prev_pose.north_m)
        prev_pose = pose

    # 1b. Compare estimated speeds on GT-stationary objects (world displacement < 0.8 m/s)
    gt_world_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
    gt_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    for raw in label_lines:
        parts = raw.strip().split()
        if len(parts) < 17:
            continue
        fr, tid, cname = int(parts[0]), int(parts[1]), parts[2]
        if cname == "DontCare" or tid < 0:
            continue
        gx, gz = float(parts[13]), float(parts[15])
        if gz <= 0.5:
            continue
        pose = oxts.get_pose(fr)
        if pose is None:
            continue
        ew, nw = pose.ego_to_world_xy(gx, gz)
        gt_world_pos[(tid, fr)] = (ew, nw)
        gt_by_frame[fr].append({"gt_tid": tid, "x_m": gx, "y_m": gz, "bbox": tuple(map(float, parts[6:10]))})

    stationary_tids = set()
    by_tid: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for (tid, fr), (ew, nw) in sorted(gt_world_pos.items()):
        by_tid[tid].append((fr, ew, nw))
    for tid, pts in by_tid.items():
        if len(pts) < 10:
            continue
        dt_span = (pts[-1][0] - pts[0][0]) / fps
        if dt_span <= 0.5:
            continue
        world_disp = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
        if world_disp / dt_span < 0.8:
            stationary_tids.add(tid)

    tl_comp = run_world_model_pass(frames, dets_by_frame, calib, oxts=oxts, fps=fps)
    tl_nocomp = run_world_model_pass(frames, dets_by_frame, calib, oxts=None, fps=fps)

    # Measure speed error on stationary objects: world-compensated speed should be near 0,
    # whereas uncompensated ego-relative speed mirrors Ego's speed!
    comp_stationary_speeds: List[float] = []
    nocomp_stationary_speeds: List[float] = []

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    for wf_c, wf_nc in zip(tl_comp, tl_nocomp):
        fr = int(wf_c["frame_idx"])
        det_map = {int(d["track_id"]): d for d in dets_by_frame.get(fr, []) if int(d["track_id"]) >= 0}
        gt_objs = gt_by_frame.get(fr, [])
        nc_map = {int(o["track_id"]): o for o in wf_nc["present_world"]["objects"]}

        for obj in wf_c["present_world"]["objects"]:
            if obj["observation_state"] != "Observed" or obj.get("speed_mps") is None:
                continue
            tid = int(obj["track_id"])
            det = det_map.get(tid)
            if det is None:
                continue
            dbox = (det["x1"], det["y1"], det["x2"], det["y2"])
            best_gt = max(gt_objs, key=lambda g: iou(dbox, g["bbox"]), default=None)
            if best_gt is None or iou(dbox, best_gt["bbox"]) < 0.4:
                continue
            if best_gt["gt_tid"] in stationary_tids:
                comp_stationary_speeds.append(float(obj["speed_mps"]))
                nc_obj = nc_map.get(tid)
                if nc_obj and nc_obj.get("speed_mps") is not None:
                    nocomp_stationary_speeds.append(float(nc_obj["speed_mps"]))

    return {
        "ego_total_distance_m": round(ego_path_len, 2),
        "max_roundtrip_transform_error_m": float(f"{max_rt_err:.3e}"),
        "stationary_gt_tracks_identified": len(stationary_tids),
        "stationary_object_samples": len(comp_stationary_speeds),
        "compensated_world_speed_on_stationary_mps": {
            "mean": round(float(np.mean(comp_stationary_speeds)), 3) if comp_stationary_speeds else None,
            "median": round(float(np.median(comp_stationary_speeds)), 3) if comp_stationary_speeds else None,
        },
        "uncompensated_ego_speed_on_stationary_mps": {
            "mean": round(float(np.mean(nocomp_stationary_speeds)), 3) if nocomp_stationary_speeds else None,
            "median": round(float(np.median(nocomp_stationary_speeds)), 3) if nocomp_stationary_speeds else None,
        },
    }


def verify_future_prediction_alignment(
    frames: Sequence[int],
    dets_by_frame: Dict[int, List[Dict]],
    calib: KittiCalibration,
    oxts: KittiOxtsOdometry,
    label_lines: Sequence[str],
    fps: float = 10.0,
) -> Dict:
    """
    RQ2: Ablate Future Prediction models against KITTI 3D GT + self-consistent future observations.
    """
    tl_oxts = run_world_model_pass(frames, dets_by_frame, calib, oxts=oxts, fps=fps)
    tl_no_oxts = run_world_model_pass(frames, dets_by_frame, calib, oxts=None, fps=fps)

    # Zero-velocity baseline: overwrite future_predictions with current position
    tl_zero_vel = []
    for wf in tl_no_oxts:
        wf_copy = json.loads(json.dumps(wf))
        for obj in wf_copy["present_world"]["objects"]:
            for fp in obj.get("future_predictions", []):
                fp["x_m"] = obj["x_m"]
                fp["y_m"] = obj["y_m"]
                fp["future_ego_x_m"] = obj["x_m"]
                fp["future_ego_y_m"] = obj["y_m"]
        tl_zero_vel.append(wf_copy)

    eval_oxts = evaluate_world_model_against_kitti_gt(tl_oxts, dets_by_frame, label_lines, oxts=oxts, fps=fps)
    eval_no_oxts = evaluate_world_model_against_kitti_gt(tl_no_oxts, dets_by_frame, label_lines, oxts=oxts, fps=fps)
    eval_zero = evaluate_world_model_against_kitti_gt(tl_zero_vel, dets_by_frame, label_lines, oxts=oxts, fps=fps)

    # Self-consistency replay: compare Predicted_{t -> t+h} (in World ENU) vs Observed_{t+h} (in World ENU)
    obs_world_by_tid_frame: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for wf in tl_oxts:
        fr = int(wf["frame_idx"])
        for obj in wf["present_world"]["objects"]:
            if obj["observation_state"] == "Observed" and obj.get("world_east_m") is not None:
                obs_world_by_tid_frame[(int(obj["track_id"]), fr)] = (float(obj["world_east_m"]), float(obj["world_north_m"]))

    replay_err_by_h: Dict[float, List[float]] = {h: [] for h in HORIZONS_SEC}
    for wf in tl_oxts:
        fr = int(wf["frame_idx"])
        for obj in wf["present_world"]["objects"]:
            if obj["observation_state"] != "Observed":
                continue
            tid = int(obj["track_id"])
            for fp in obj.get("future_predictions", []):
                if fp.get("world_east_m") is None:
                    continue
                h = float(fp["horizon_sec"])
                fut_fr = fr + int(round(h * fps))
                obs_fut = obs_world_by_tid_frame.get((tid, fut_fr))
                if obs_fut is not None:
                    replay_err_by_h[h].append(math.hypot(float(fp["world_east_m"]) - obs_fut[0], float(fp["world_north_m"]) - obs_fut[1]))

    return {
        "oxts_world_kalman_with_ctrv": eval_oxts,
        "ego_relative_kalman_no_oxts": eval_no_oxts["motion_prediction_metrics"],
        "zero_velocity_baseline": eval_zero["motion_prediction_metrics"],
        "self_consistency_world_replay_error_m": {
            str(h): {
                "samples": len(vals),
                "mean_error_m": round(float(np.mean(vals)), 4) if vals else None,
            }
            for h, vals in replay_err_by_h.items()
        },
    }


def _perturb_detections(
    dets_by_frame: Dict[int, List[Dict]],
    frames: Sequence[int],
    seed: int = 42,
    conf_scale: float = 0.45,
    bbox_jitter_px: float = 6.0,
    dropout_prob: float = 0.30,
) -> Dict[int, List[Dict]]:
    rng = random.Random(seed)
    perturbed: Dict[int, List[Dict]] = {}
    for fr in frames:
        out = []
        for d in dets_by_frame.get(fr, []):
            if rng.random() < dropout_prob:
                continue
            nd = dict(d)
            nd["conf"] = round(max(0.05, float(d["conf"]) * conf_scale), 4)
            dx = rng.uniform(-bbox_jitter_px, bbox_jitter_px)
            dy = rng.uniform(-bbox_jitter_px, bbox_jitter_px)
            nd["x1"] = float(d["x1"]) + dx
            nd["x2"] = float(d["x2"]) + dx
            nd["y1"] = float(d["y1"]) + dy
            nd["y2"] = float(d["y2"]) + dy
            out.append(nd)
        perturbed[fr] = out
    return perturbed


def verify_uncertainty_causality(
    frames: Sequence[int],
    dets_by_frame: Dict[int, List[Dict]],
    calib: KittiCalibration,
    oxts: KittiOxtsOdometry,
    fps: float = 10.0,
) -> Dict:
    """
    RQ3: Verify that higher perception uncertainty causally changes Risk Indicator & Safety Shield.
    Compares:
      - Arm A: Uncertainty-Blind (uncertainty_aware=False)
      - Arm B: Uncertainty-Aware Nominal (clean detections)
      - Arm C: Degraded Perception (lower confidence + bbox jitter + 30% occlusion dropouts)
    """
    tl_blind = run_world_model_pass(frames, dets_by_frame, calib, oxts=oxts, fps=fps, uncertainty_aware=False)
    tl_nominal = run_world_model_pass(frames, dets_by_frame, calib, oxts=oxts, fps=fps, uncertainty_aware=True)
    degraded_dets = _perturb_detections(dets_by_frame, frames)
    tl_degraded = run_world_model_pass(frames, degraded_dets, calib, oxts=oxts, fps=fps, uncertainty_aware=True)

    def summarize_arm(tl: List[Dict]) -> Dict:
        risks = [float(w["risk_assessment"]["max_risk_indicator"]) for w in tl]
        decs = [w["risk_assessment"]["safety_shield_decision"] for w in tl]
        counts = Counter(decs)
        non_normal = sum(1 for d in decs if d != "Normal Driving")
        return {
            "frames": len(tl),
            "mean_risk_indicator": round(float(np.mean(risks)), 4),
            "max_risk_indicator": round(float(np.max(risks)), 4),
            "non_normal_intervention_rate": round(non_normal / max(1, len(tl)), 4),
            "decision_counts": {lvl: counts.get(lvl, 0) for lvl in SAFETY_LEVELS},
        }

    esc_nom_over_blind = sum(
        1
        for a, b in zip(tl_blind, tl_nominal)
        if SAFETY_RANK[b["risk_assessment"]["safety_shield_decision"]]
        > SAFETY_RANK[a["risk_assessment"]["safety_shield_decision"]]
    )
    esc_deg_over_nom = sum(
        1
        for b, c in zip(tl_nominal, tl_degraded)
        if SAFETY_RANK[c["risk_assessment"]["safety_shield_decision"]]
        > SAFETY_RANK[b["risk_assessment"]["safety_shield_decision"]]
    )
    risk_inc_deg_over_nom = sum(
        1
        for b, c in zip(tl_nominal, tl_degraded)
        if c["risk_assessment"]["max_risk_indicator"] > b["risk_assessment"]["max_risk_indicator"]
    )

    arm_blind = summarize_arm(tl_blind)
    arm_nom = summarize_arm(tl_nominal)
    arm_deg = summarize_arm(tl_degraded)

    return {
        "arm_a_uncertainty_blind": arm_blind,
        "arm_b_uncertainty_aware_nominal": arm_nom,
        "arm_c_uncertainty_aware_degraded": arm_deg,
        "causality_deltas": {
            "mean_risk_shift_aware_minus_blind": round(arm_nom["mean_risk_indicator"] - arm_blind["mean_risk_indicator"], 4),
            "mean_risk_shift_degraded_minus_nominal": round(arm_deg["mean_risk_indicator"] - arm_nom["mean_risk_indicator"], 4),
            "frames_escalated_aware_over_blind": esc_nom_over_blind,
            "frames_escalated_degraded_over_nominal": esc_deg_over_nom,
            "frames_risk_increased_degraded_over_nominal": risk_inc_deg_over_nom,
            "total_frames": len(frames),
        },
    }


def verify_collision_detector_smoothness(
    frames: Sequence[int],
    dets_by_frame: Dict[int, List[Dict]],
    calib: KittiCalibration,
    oxts: KittiOxtsOdometry,
    fps: float = 10.0,
) -> Dict:
    """
    Compare Unsmoothed raw hazard spikes vs C1/slew-rate Gradient-Smoothed collision risk.
    """
    tl_raw = run_world_model_pass(
        frames, dets_by_frame, calib, oxts=oxts, fps=fps,
        uncertainty_aware=True, enable_risk_smoothing=False,
    )
    tl_smooth = run_world_model_pass(
        frames, dets_by_frame, calib, oxts=oxts, fps=fps,
        uncertainty_aware=True, enable_risk_smoothing=True,
    )

    def summarize_dynamics(tl: List[Dict], use_raw_key: bool) -> Dict:
        risks = [
            float(w["risk_assessment"]["raw_max_risk_indicator" if use_raw_key else "max_risk_indicator"])
            for w in tl
        ]
        times = [float(w["timestamp_sec"]) for w in tl]
        decs = [w["risk_assessment"]["safety_shield_decision"] for w in tl]
        grads: List[float] = []
        for idx in range(1, len(risks)):
            dt = max(1e-3, times[idx] - times[idx - 1])
            grads.append(abs(risks[idx] - risks[idx - 1]) / dt)
        flips = sum(1 for idx in range(1, len(decs)) if decs[idx] != decs[idx - 1])
        spikes = sum(1 for g in grads if g > 1.5)
        return {
            "mean_abs_grad_per_sec": round(float(np.mean(grads)), 4) if grads else 0.0,
            "p95_abs_grad_per_sec": round(float(np.percentile(grads, 95)), 4) if grads else 0.0,
            "max_abs_grad_per_sec": round(float(np.max(grads)), 4) if grads else 0.0,
            "spike_count_gt_1p5_per_sec": int(spikes),
            "decision_flip_count": int(flips),
            "mean_risk": round(float(np.mean(risks)), 4) if risks else 0.0,
        }

    unsmoothed = summarize_dynamics(tl_raw, use_raw_key=True)
    smoothed = summarize_dynamics(tl_smooth, use_raw_key=False)

    def pct_reduction(before: float, after: float) -> Optional[float]:
        if before <= 1e-6:
            return None
        return round((before - after) / before * 100.0, 2)

    return {
        "unsmoothed": unsmoothed,
        "gradient_smoothed": smoothed,
        "reduction": {
            "mean_grad_reduction_pct": pct_reduction(
                unsmoothed["mean_abs_grad_per_sec"], smoothed["mean_abs_grad_per_sec"]
            ),
            "p95_grad_reduction_pct": pct_reduction(
                unsmoothed["p95_abs_grad_per_sec"], smoothed["p95_abs_grad_per_sec"]
            ),
            "max_grad_reduction_pct": pct_reduction(
                unsmoothed["max_abs_grad_per_sec"], smoothed["max_abs_grad_per_sec"]
            ),
            "spike_reduction_pct": pct_reduction(
                float(unsmoothed["spike_count_gt_1p5_per_sec"]),
                float(smoothed["spike_count_gt_1p5_per_sec"]),
            ),
            "decision_flip_reduction_pct": pct_reduction(
                float(unsmoothed["decision_flip_count"]),
                float(smoothed["decision_flip_count"]),
            ),
        },
    }


def run_all_validations(
    seq_id: str,
    trajectories_csv: Path,
    calib_file: Path,
    oxts_file: Path,
    label_file: Path,
    fps: float = 10.0,
) -> Dict:
    calib = KittiCalibration.from_calib_file(calib_file, seq_id=seq_id)
    oxts = KittiOxtsOdometry.from_oxts_file(oxts_file)
    dets_by_frame = load_trajectories_csv(trajectories_csv)
    label_lines = label_file.read_text(encoding="utf-8").splitlines()
    frames = sorted(oxts.poses.keys())

    return {
        "sequence_id": seq_id,
        "frame_count": len(frames),
        "rq1_coordinate_and_ego_motion": verify_coordinate_transforms_and_ego_compensation(
            frames, dets_by_frame, calib, oxts, label_lines, fps=fps
        ),
        "rq2_future_prediction_vs_observation": verify_future_prediction_alignment(
            frames, dets_by_frame, calib, oxts, label_lines, fps=fps
        ),
        "rq3_uncertainty_causality_on_safety": verify_uncertainty_causality(
            frames, dets_by_frame, calib, oxts, fps=fps
        ),
        "rq4_collision_detector_gradient_smoothing": verify_collision_detector_smoothness(
            frames, dets_by_frame, calib, oxts, fps=fps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1-RQ3 Scientific Validation Experiments")
    parser.add_argument("--seq-id", type=str, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--calib-file", type=Path, required=True)
    parser.add_argument("--oxts-file", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    res = run_all_validations(
        seq_id=args.seq_id,
        trajectories_csv=args.trajectories_csv,
        calib_file=args.calib_file,
        oxts_file=args.oxts_file,
        label_file=args.label_file,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

