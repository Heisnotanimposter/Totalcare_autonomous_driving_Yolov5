#!/usr/bin/env python3
"""
Split-Screen Visualizer for Camera-Centric Temporal Ego-centric World Model.

Left Panel  : Observed World (raw camera + 2D detections, class, track ID, confidence)
Right Panel : Inferred World (Ego-centric BEV with Past/Present/Future states,
              Observed/Inferred/Occluded/Unknown separation, uncertainty ellipses,
              Risk Indicator, and advisory Safety Shield Decision)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from track_utils import inspect_image_directory
from world_model import KittiCalibration, KittiOxtsOdometry, TemporalWorldModel, evaluate_world_model_against_kitti_gt


CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Car": (80, 220, 80),
    "Pedestrian": (0, 200, 255),
    "Cyclist": (255, 180, 40),
    "Van": (140, 210, 140),
    "Truck": (60, 160, 220),
}

SAFETY_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Normal Driving": (70, 180, 70),
    "Caution": (0, 200, 230),
    "Speed Reduction": (0, 140, 255),
    "Braking Recommendation": (40, 70, 240),
    "Emergency Intervention": (30, 30, 220),
}


def _stage_for_time(t_sec: float, staged_demo: bool) -> Tuple[int, str]:
    if not staged_demo or t_sec >= 4.0:
        return 5, "Stage 4-5s+: Risk Assessment & Safety Shield (Full Pipeline)"
    if t_sec < 1.0:
        return 1, "Stage 0-1s: Raw Observation & Ego Reference Init"
    if t_sec < 2.0:
        return 2, "Stage 1-2s: Object Perception (2D Det + Track ID)"
    if t_sec < 3.0:
        return 3, "Stage 2-3s: Temporal Understanding (Past + Present World)"
    return 4, "Stage 3-4s: Future Prediction (Horizons + Uncertainty)"


class BevCanvasMapper:
    """Maps Ego-centric BEV meters (X right, Y forward) to pixel coordinates (u, v)."""

    def __init__(
        self,
        width: int = 640,
        height: int = 540,
        x_range: Tuple[float, float] = (-18.0, 18.0),
        y_range: Tuple[float, float] = (-10.0, 52.0),
    ):
        self.width = width
        self.height = height
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def to_px(self, x_m: float, y_m: float) -> Tuple[int, int]:
        u = int(round((x_m - self.x_min) / (self.x_max - self.x_min) * (self.width - 1)))
        v = int(round((self.y_max - y_m) / (self.y_max - self.y_min) * (self.height - 1)))
        return u, v

    def scale_px_per_m(self) -> Tuple[float, float]:
        sx = (self.width - 1) / (self.x_max - self.x_min)
        sy = (self.height - 1) / (self.y_max - self.y_min)
        return sx, sy


def _match_mask_polygon(det: Dict, frame_mask_entries: Sequence[Dict]) -> Optional[List[List[int]]]:
    if not frame_mask_entries:
        return None
    tid = int(det.get("track_id", -1))
    if tid >= 0:
        for entry in frame_mask_entries:
            if int(entry.get("track_id", -2)) == tid:
                return entry.get("polygon_xy")
    dx1, dy1, dx2, dy2 = float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
    best_poly = None
    best_dist = 12.0
    for entry in frame_mask_entries:
        bbox = entry.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        dist = abs(dx1 - bbox[0]) + abs(dy1 - bbox[1]) + abs(dx2 - bbox[2]) + abs(dy2 - bbox[3])
        if dist < best_dist:
            best_dist = dist
            best_poly = entry.get("polygon_xy")
    return best_poly


def render_left_observed_panel(
    bgr_frame: np.ndarray,
    frame_dets: Sequence[Dict],
    stage: int,
    target_size: Tuple[int, int],
    frame_masks: Optional[Sequence[Dict]] = None,
) -> np.ndarray:
    """Render Left Panel (Observed World) strictly using directly observed 2D detections + instance masks."""
    tw, th = target_size
    fh, fw = bgr_frame.shape[:2]
    scale = min(tw / fw, th / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_AREA)

    panel = np.zeros((th, tw, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2

    vis = resized.copy()
    if stage >= 2:
        mask_layer = vis.copy()
        scaled_polys: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        if frame_masks:
            for d in frame_dets:
                poly = _match_mask_polygon(d, frame_masks)
                if poly and len(poly) >= 3:
                    cname = str(d["class_name"])
                    color = CLASS_COLORS_BGR.get(cname, (200, 200, 200))
                    pts_scaled = np.round(np.asarray(poly, dtype=np.float32) * scale).astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_layer, [pts_scaled], color)
                    scaled_polys.append((pts_scaled, color))
        if scaled_polys:
            vis = cv2.addWeighted(mask_layer, 0.38, vis, 0.62, 0.0)
            for pts_scaled, color in scaled_polys:
                cv2.polylines(vis, [pts_scaled], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        for d in frame_dets:
            x1 = int(round(float(d["x1"]) * scale))
            y1 = int(round(float(d["y1"]) * scale))
            x2 = int(round(float(d["x2"]) * scale))
            y2 = int(round(float(d["y2"]) * scale))
            cname = str(d["class_name"])
            tid = int(d["track_id"])
            conf = float(d["conf"])
            color = CLASS_COLORS_BGR.get(cname, (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1 if scaled_polys else 2)
            label = f"{cname} #{tid} {conf:.2f}" if tid >= 0 else f"{cname} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

    panel[oy : oy + nh, ox : ox + nw] = vis

    # Header badge
    cv2.rectangle(panel, (0, 0), (tw, 34), (28, 32, 38), -1)
    header_txt = "LEFT PANEL | OBSERVED WORLD (Camera 2D + Instance Masks)" if frame_masks else "LEFT PANEL | OBSERVED WORLD (Front Camera 2D)"
    cv2.putText(
        panel,
        header_txt,
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (235, 242, 250),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_right_world_panel(
    world_state: Dict,
    stage: int,
    panel_size: Tuple[int, int],
) -> np.ndarray:
    """Render Right Panel (Inferred Ego-centric World Model BEV)."""
    pw, ph = panel_size
    mapper = BevCanvasMapper(width=pw, height=ph)
    sx, sy = mapper.scale_px_per_m()

    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    # 1. Base Unknown region (outside front camera FOV & rear area)
    panel[:] = (26, 26, 32)
    # Subtle diagonal hatching to visually distinguish Unknown space
    for diag in range(-ph, pw + ph, 22):
        cv2.line(panel, (diag, 0), (diag - ph, ph), (36, 36, 44), 1)

    # 2. Forward Camera FOV cone (Observed / Monitored sector)
    fov_deg = world_state.get("fov_deg") or 81.0
    half_fov = math.radians(fov_deg / 2.0)
    max_r = 55.0
    ego_px = mapper.to_px(0.0, 0.0)
    fov_pts = [ego_px]
    for k in range(25):
        ang = -half_fov + (2.0 * half_fov * k / 24.0)
        fov_pts.append(mapper.to_px(max_r * math.sin(ang), max_r * math.cos(ang)))
    fov_poly = np.array(fov_pts, dtype=np.int32)
    cv2.fillPoly(panel, [fov_poly], (34, 44, 52))
    cv2.polylines(panel, [fov_poly], True, (75, 110, 130), 1, cv2.LINE_AA)

    # Range rings (10m..50m) inside FOV
    for r_m in (10, 20, 30, 40, 50):
        arc_pts = []
        for k in range(25):
            ang = -half_fov + (2.0 * half_fov * k / 24.0)
            arc_pts.append(mapper.to_px(r_m * math.sin(ang), r_m * math.cos(ang)))
        cv2.polylines(panel, [np.array(arc_pts, dtype=np.int32)], False, (52, 66, 78), 1, cv2.LINE_AA)
        lx, ly = mapper.to_px(r_m * math.sin(half_fov) * 0.88, r_m * math.cos(half_fov) * 0.88)
        cv2.putText(panel, f"{r_m}m", (lx - 26, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (125, 145, 160), 1)

    # 3. Road Structure (nominal lane geometry) & Ego Planned Path
    lane_hw = float(world_state["ego"]["corridor_half_width_m"])
    for x_lane, col in ((-lane_hw * 3.0, (70, 75, 85)), (-lane_hw, (120, 135, 150)), (lane_hw, (120, 135, 150)), (lane_hw * 3.0, (70, 75, 85))):
        p_bot = mapper.to_px(x_lane, 0.0)
        p_top = mapper.to_px(x_lane, 50.0)
        cv2.line(panel, p_bot, p_top, col, 1, cv2.LINE_AA)

    if stage >= 4:
        corr_poly = np.array(
            [
                mapper.to_px(-lane_hw, 0.0),
                mapper.to_px(lane_hw, 0.0),
                mapper.to_px(lane_hw, 35.0),
                mapper.to_px(-lane_hw, 35.0),
            ],
            dtype=np.int32,
        )
        overlay = panel.copy()
        cv2.fillPoly(overlay, [corr_poly], (45, 85, 95))
        cv2.addWeighted(overlay, 0.30, panel, 0.70, 0, panel)

        ego_pts = [mapper.to_px(pt["x_m"], pt["y_m"]) for pt in world_state["future_world"]["ego_planned_path"]]
        cv2.polylines(panel, [np.array(ego_pts, dtype=np.int32)], False, (255, 210, 90), 2, cv2.LINE_AA)

    # 4. Occluded sectors behind foreground objects
    if stage >= 2:
        occ_overlay = panel.copy()
        for poly_m in world_state["present_world"]["occlusion_sectors"]:
            pts = np.array([mapper.to_px(px, py) for px, py in poly_m], dtype=np.int32)
            cv2.fillPoly(occ_overlay, [pts], (18, 20, 26))
        cv2.addWeighted(occ_overlay, 0.65, panel, 0.35, 0, panel)

    # 5. Past World expired tracks (fading memory)
    if stage >= 3:
        for exp in world_state["past_world"]["expired_tracks"]:
            hist = exp.get("history", [])
            if len(hist) >= 2:
                pts = np.array([mapper.to_px(p["x_m"], p["y_m"]) for p in hist], dtype=np.int32)
                cv2.polylines(panel, [pts], False, (90, 95, 105), 1, cv2.LINE_AA)

    # 6. Present + Future World Objects
    if stage >= 2:
        for obj in world_state["present_world"]["objects"]:
            x_m = float(obj["x_m"])
            y_m = float(obj["y_m"])
            tid = int(obj["track_id"])
            cname = str(obj["class_name"])
            obs_state = obj["observation_state"]
            base_col = CLASS_COLORS_BGR.get(cname, (200, 200, 200))
            risk = float(obj.get("risk_indicator", 0.0))

            if stage >= 5 and risk >= 0.45:
                draw_col = (50, 70, 255)
            elif obs_state == "Occluded":
                draw_col = (180, 140, 240)
            else:
                draw_col = base_col

            # Past World trajectory trail (Stage 3+)
            if stage >= 3 and len(obj.get("past_trail", [])) >= 2:
                trail = obj["past_trail"]
                for i in range(1, len(trail)):
                    w = float(trail[i]["temporal_weight"])
                    p0 = mapper.to_px(trail[i - 1]["x_m"], trail[i - 1]["y_m"])
                    p1 = mapper.to_px(trail[i]["x_m"], trail[i]["y_m"])
                    tcol = tuple(int(c * max(0.25, w)) for c in base_col)
                    cv2.line(panel, p0, p1, tcol, 2, cv2.LINE_AA)

            # Future World multi-horizon prediction & covariance ellipses (Stage 4+)
            if stage >= 4 and obj.get("future_predictions"):
                fut_pts = [mapper.to_px(x_m, y_m)]
                for fp in obj["future_predictions"]:
                    fx, fy = float(fp["x_m"]), float(fp["y_m"])
                    fpx = mapper.to_px(fx, fy)
                    fut_pts.append(fpx)
                    ax_maj = max(2, int(round(float(fp["sigma_major_m"]) * sx)))
                    ax_min = max(2, int(round(float(fp["sigma_minor_m"]) * sy)))
                    ang = -float(fp["ellipse_angle_deg"])
                    cv2.ellipse(panel, fpx, (ax_maj, ax_min), ang, 0, 360, (180, 120, 255), 1, cv2.LINE_AA)
                    cv2.circle(panel, fpx, 3, (220, 170, 255), -1)
                for a, b in zip(fut_pts, fut_pts[1:]):
                    cv2.line(panel, a, b, (210, 140, 255), 1, cv2.LINE_AA)

            # Present object box in BEV
            cx_px, cy_px = mapper.to_px(x_m, y_m)
            box_w = max(8, int(round((1.8 if cname == "Car" else 0.7) * sx)))
            box_h = max(10, int(round((4.0 if cname == "Car" else 0.8) * sy)))
            tl = (cx_px - box_w // 2, cy_px - box_h // 2)
            br = (cx_px + box_w // 2, cy_px + box_h // 2)
            if obs_state == "Observed":
                cv2.rectangle(panel, tl, br, draw_col, -1)
                cv2.rectangle(panel, tl, br, (255, 255, 255), 1)
            else:
                # Occluded / Inferred coasting object rendered hollow/dashed-style
                cv2.rectangle(panel, tl, br, draw_col, 2)

            # Calibrated velocity vector (Stage 3+)
            if stage >= 3 and obj.get("vx_mps") is not None and obj.get("vy_mps") is not None:
                vx, vy = float(obj["vx_mps"]), float(obj["vy_mps"])
                if math.hypot(vx, vy) > 0.4:
                    tip = mapper.to_px(x_m + vx * 0.5, y_m + vy * 0.5)
                    cv2.arrowedLine(panel, (cx_px, cy_px), tip, (255, 255, 180), 2, tipLength=0.25)

            lbl = f"#{tid} {y_m:.1f}m"
            if stage >= 3 and obj.get("speed_mps") is not None:
                lbl += f" {obj['speed_mps']:.1f}m/s"
            if obs_state != "Observed":
                lbl += " [INF]"
            cv2.putText(panel, lbl, (br[0] + 4, cy_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (235, 240, 245), 1, cv2.LINE_AA)

    # 7. Future conflict zone markers (Stage 4-5)
    if stage >= 4:
        for cz in world_state["future_world"]["conflict_zones"]:
            cpx = mapper.to_px(cz["x_m"], cz["y_m"])
            rad = max(8, int(round(cz["radius_m"] * sx)))
            cv2.circle(panel, cpx, rad, (40, 60, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"CONFLICT +{cz['horizon_sec']}s", (cpx[0] + rad + 2, cpx[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 130, 255), 1)

    # 8. Ego Vehicle past trail & icon at (0, 0)
    ego_past = world_state["ego"].get("past_trail", [])
    if stage >= 3 and len(ego_past) >= 2:
        ep_pts = np.array([mapper.to_px(p["x_m"], p["y_m"]) for p in ego_past], dtype=np.int32)
        cv2.polylines(panel, [ep_pts], False, (120, 170, 200), 1, cv2.LINE_AA)
    ex, ey = ego_px
    ego_pts = np.array([(ex, ey - 14), (ex - 10, ey + 10), (ex + 10, ey + 10)], dtype=np.int32)
    cv2.fillPoly(panel, [ego_pts], (255, 220, 60))
    cv2.polylines(panel, [ego_pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "EGO (0,0)", (ex - 28, ey + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 120), 1, cv2.LINE_AA)

    # Explicitly label rear UNKNOWN region
    cv2.putText(panel, "UNKNOWN (Outside Front Camera FOV)", (ex - 130, ph - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (145, 150, 165), 1, cv2.LINE_AA)

    # Legend box
    cv2.rectangle(panel, (10, 42), (218, 132), (20, 24, 30), -1)
    cv2.rectangle(panel, (10, 42), (218, 132), (70, 80, 95), 1)
    cv2.putText(panel, "Observed (Solid)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 230, 110), 1)
    cv2.putText(panel, "Inferred / Occluded (Hollow)", (20, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 150, 245), 1)
    cv2.putText(panel, "Future + 1-sigma Ellipses", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 160, 255), 1)
    cv2.putText(panel, "Unknown (Hatched Rear/Side)", (20, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 170), 1)
    cv2.putText(panel, "Ego Corridor & Planned Path", (20, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 215, 95), 1)

    # Header badge
    cv2.rectangle(panel, (0, 0), (pw, 34), (28, 32, 38), -1)
    cv2.putText(
        panel,
        "RIGHT PANEL | INFERRED EGO-CENTRIC WORLD MODEL (BEV)",
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (235, 242, 250),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_telemetry_hud(
    world_state: Dict,
    stage: int,
    stage_title: str,
    total_width: int,
    hud_height: int = 92,
) -> np.ndarray:
    """Render bottom telemetry bar with Temporal State, Uncertainties, Risk Indicator & Safety Shield."""
    hud = np.zeros((hud_height, total_width, 3), dtype=np.uint8)
    hud[:] = (22, 25, 31)
    cv2.line(hud, (0, 0), (total_width, 0), (70, 82, 96), 1)

    ts = world_state["timestamp_sec"]
    fr = world_state["frame_idx"]
    objs = world_state["present_world"]["objects"]
    obs_cnt = sum(1 for o in objs if o["observation_state"] == "Observed")
    occ_cnt = sum(1 for o in objs if o["observation_state"] != "Observed")

    cv2.putText(
        hud,
        f"Frame {fr:04d} | t = {ts:.2f}s | {stage_title}",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (240, 245, 250),
        1,
        cv2.LINE_AA,
    )

    if objs:
        avg_det_u = sum(o["uncertainties"]["detection_uncertainty"] for o in objs) / len(objs)
        avg_dep_u = sum(o["uncertainties"]["depth_uncertainty_m"] for o in objs) / len(objs)
        avg_trk_u = sum(o["uncertainties"]["tracking_uncertainty"] for o in objs) / len(objs)
        pred_vals = [o["uncertainties"]["motion_prediction_uncertainty_2s_m"] for o in objs if o["uncertainties"]["motion_prediction_uncertainty_2s_m"] is not None]
        avg_prd_u = f"{sum(pred_vals) / len(pred_vals):.2f}m" if pred_vals else "N/A"
        unc_text = f"Uncertainties -> Det:{avg_det_u:.2f} | Depth:+/-{avg_dep_u:.2f}m | Track:{avg_trk_u:.2f} | Pred(+2s):+/-{avg_prd_u}"
    else:
        unc_text = "Uncertainties -> No active calibrated tracks in FOV"

    cv2.putText(
        hud,
        f"Tracks: Observed={obs_cnt}, Occluded/Inferred={occ_cnt} | {unc_text}",
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (185, 200, 215),
        1,
        cv2.LINE_AA,
    )
    ego_comp_str = "OXTS Ego-Comp ON" if world_state.get("ego_motion_compensated") else "Ego-Comp OFF"
    ego_spd = world_state["ego"].get("speed_mps")
    spd_str = f" | Ego v={ego_spd:.1f}m/s" if ego_spd is not None else ""
    cv2.putText(
        hud,
        f"Coord: Ego-centric ({ego_comp_str}{spd_str}) | Note: Risk Indicator is heuristic, NOT collision probability.",
        (16, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (145, 160, 175),
        1,
        cv2.LINE_AA,
    )

    # Right-side Risk Indicator & Safety Shield Decision box
    risk = world_state["risk_assessment"]
    r_val = float(risk["max_risk_indicator"]) if stage >= 5 else 0.0
    ttc_val = risk.get("min_calibrated_ttc_sec") if stage >= 5 else None
    decision = risk["safety_shield_decision"] if stage >= 5 else "Initializing..."
    d_col = SAFETY_COLORS_BGR.get(decision, (100, 120, 140))

    box_x1 = total_width - 442
    cv2.rectangle(hud, (box_x1, 10), (total_width - 14, hud_height - 10), (32, 36, 44), -1)
    cv2.rectangle(hud, (box_x1, 10), (total_width - 14, hud_height - 10), d_col, 2)

    ttc_str = f"{ttc_val:.2f}s" if ttc_val is not None else "N/A"
    smooth_tag = " | Smoothed" if risk.get("risk_smoothing_enabled") else ""
    cv2.putText(
        hud,
        f"Risk Indicator: {r_val:.2f} | Calib TTC: {ttc_str}{smooth_tag}",
        (box_x1 + 12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        hud,
        f"Safety Shield (Advisory): {decision}",
        (box_x1 + 12, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        d_col,
        1,
        cv2.LINE_AA,
    )
    return hud


def generate_world_model_demo(
    images_dir: Path,
    trajectories_csv: Path,
    output_dir: Path,
    calib_file: Optional[Path] = None,
    label_file: Optional[Path] = None,
    oxts_file: Optional[Path] = None,
    masks_json: Optional[Path] = None,
    seq_id: str = "0019",
    fps: float = 10.0,
    max_frames: Optional[int] = 80,
    staged_demo: bool = True,
) -> Dict:
    """
    Build Temporal Ego-centric World Model timeline and render split-screen MP4 + keyframes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed_paths, gaps = inspect_image_directory(images_dir)
    if gaps:
        raise ValueError(f"Non-contiguous frames in {images_dir}: {gaps[:5]}")
    if max_frames is not None:
        indexed_paths = indexed_paths[:max_frames]

    first_img = cv2.imread(str(indexed_paths[0][1]))
    if first_img is None:
        raise RuntimeError(f"Cannot read {indexed_paths[0][1]}")
    img_h, img_w = first_img.shape[:2]

    calib = KittiCalibration.from_calib_file(calib_file, seq_id=seq_id) if (calib_file and calib_file.is_file()) else None
    oxts = KittiOxtsOdometry.from_oxts_file(oxts_file) if (oxts_file and oxts_file.is_file()) else None

    if masks_json is None:
        auto_masks = trajectories_csv.parent / "trajectories_masks.json"
        if auto_masks.is_file():
            masks_json = auto_masks
    masks_by_frame: Dict[str, List[Dict]] = {}
    mask_backend = "none"
    if masks_json is not None and masks_json.is_file():
        m_payload = json.loads(masks_json.read_text(encoding="utf-8"))
        masks_by_frame = m_payload.get("masks_by_frame", {})
        mask_backend = m_payload.get("mask_backend", "json")

    dets_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    with open(trajectories_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sfr = int(row.get("source_frame_idx", row["frame_idx"]))
            row["track_id"] = int(row["track_id"])
            for k in ("conf", "x1", "y1", "x2", "y2"):
                row[k] = float(row[k])
            dets_by_frame[sfr].append(row)

    world = TemporalWorldModel(calibration=calib, oxts=oxts, image_width=img_w, image_height=img_h)
    left_size = (720, 540)
    right_size = (640, 540)
    hud_h = 92
    total_w = left_size[0] + right_size[0]
    total_h = left_size[1] + hud_h

    video_path = output_dir / f"world_model_demo_{seq_id}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {video_path}")

    world_timeline: List[Dict] = []
    keyframe_targets = {5, 15, 25, 35, 45, 55}
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    base_sfr = indexed_paths[0][0]
    try:
        for idx, (sfr, img_path) in enumerate(indexed_paths):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            t_sec = (sfr - base_sfr) / fps
            frame_dets = dets_by_frame.get(sfr, [])
            w_state = world.step(frame_idx=sfr, timestamp_sec=t_sec, detections=frame_dets)
            world_timeline.append(w_state)

            stage, stage_title = _stage_for_time(t_sec, staged_demo=staged_demo)
            frame_masks = masks_by_frame.get(str(sfr))
            left = render_left_observed_panel(bgr, frame_dets, stage=stage, target_size=left_size, frame_masks=frame_masks)
            right = render_right_world_panel(w_state, stage=stage, panel_size=right_size)
            hud = render_telemetry_hud(w_state, stage=stage, stage_title=stage_title, total_width=total_w, hud_height=hud_h)

            top = np.hstack([left, right])
            composite = np.vstack([top, hud])
            writer.write(composite)

            if idx in keyframe_targets:
                cv2.imwrite(str(keyframes_dir / f"seq_{seq_id}_frame_{sfr:06d}_stage{stage}.png"), composite)
    finally:
        writer.release()

    timeline_path = output_dir / f"world_model_timeline_{seq_id}.json"
    timeline_path.write_text(json.dumps(world_timeline, indent=2), encoding="utf-8")

    eval_metrics = None
    if label_file is not None and label_file.is_file():
        eval_metrics = evaluate_world_model_against_kitti_gt(
            world_frames=world_timeline,
            detections_by_frame=dets_by_frame,
            label_txt_lines=label_file.read_text(encoding="utf-8").splitlines(),
            oxts=oxts,
            fps=fps,
        )
        (output_dir / f"world_model_eval_{seq_id}.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    summary = {
        "sequence_id": seq_id,
        "frames_rendered": len(world_timeline),
        "fps": fps,
        "duration_sec": round(len(world_timeline) / fps, 2),
        "calibrated": calib is not None,
        "ego_motion_compensated": oxts is not None,
        "mask_backend": mask_backend,
        "staged_demo": staged_demo,
        "video_path": str(video_path),
        "timeline_path": str(timeline_path),
        "eval_metrics": eval_metrics,
    }
    (output_dir / f"world_model_summary_{seq_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera-Centric Temporal Ego-centric World Model Visualizer")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calib-file", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--oxts-file", type=Path, default=None)
    parser.add_argument("--masks-json", type=Path, default=None)
    parser.add_argument("--seq-id", type=str, default="0019")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--full-overlay", action="store_true", help="Disable 0-5s staged intro and show all layers from t=0")
    args = parser.parse_args()

    out = generate_world_model_demo(
        images_dir=args.images_dir,
        trajectories_csv=args.trajectories_csv,
        output_dir=args.output_dir,
        calib_file=args.calib_file,
        label_file=args.label_file,
        oxts_file=args.oxts_file,
        masks_json=args.masks_json,
        seq_id=args.seq_id,
        fps=args.fps,
        max_frames=args.max_frames,
        staged_demo=not args.full_overlay,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

