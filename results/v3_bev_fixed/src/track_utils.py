#!/usr/bin/env python3
"""
Traffic Multi-Object Tracking, Center-Coordinate Export & KITTI/TrackEval Formatter.

Static-audit remediation update (PHASE 0 — experimental tooling, NOT FINAL):
  1. Separates raw Detection Records (`track_id` may be -1) from Valid Track Records
     (`track_id >= 0`): unmatched detections stay in CSV but NEVER merge into a fake `-1` track.
  2. Sorts valid track records chronologically, flags out-of-order inputs, duplicate
     `(track_id, frame_idx)` detections, and per-track class switches.
  3. Computes timestamp-based 2D image velocities (`px/sec`) alongside `px/frame` and
     distinguishes 2D image-plane motion from physical world velocity (`m/s`).
  4. Preserves `source_frame_idx`, enforces explicit FPS for image directories, checks
     VideoWriter/image-decode failures with `try/finally` cleanup, and validates
     explicit `KITTI_CLASS_MAP` mapping for 2D TrackEval predictions.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FRAME_DIGITS_RE = re.compile(r"(\d+)(?=(?:_endo)?\.[A-Za-z]+$)")

KITTI_CLASS_MAP: Dict[str, str] = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
    "van": "Van",
    "truck": "Truck",
    "person": "Person",
    "person_sitting": "Person_sitting",
    "tram": "Tram",
    "misc": "Misc",
}


def extract_source_frame_idx(path: Path, fallback_idx: int) -> int:
    """Parse numeric frame index from filenames such as `000120.png` or `frame_120_endo.png`."""
    m = FRAME_DIGITS_RE.search(path.name)
    return int(m.group(1)) if m else fallback_idx


def clip_xyxy_to_bounds(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float, bool]:
    """Clip bounding box coordinates to [0, width] x [0, height] and return whether clipping occurred."""
    cx1 = max(0.0, min(float(width), x1))
    cy1 = max(0.0, min(float(height), y1))
    cx2 = max(0.0, min(float(width), x2))
    cy2 = max(0.0, min(float(height), y2))
    clipped = (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2)
    return cx1, cy1, cx2, cy2, clipped


def compute_track_trajectories(records: Sequence[Dict]) -> Dict:
    """
    Aggregate valid (`track_id >= 0`) detection rows into temporally sorted trajectories.
    Detects out-of-order inputs, duplicate `(track_id, frame_idx)` rows, and class switches.
    """
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    untracked_count = 0
    for row in records:
        tid = int(row["track_id"])
        if tid < 0:
            untracked_count += 1
            continue
        grouped[tid].append(row)

    out_of_order_events: List[Dict] = []
    duplicate_frame_events: List[Dict] = []
    class_inconsistency_events: List[Dict] = []
    tracks_out: List[Dict] = []

    for tid in sorted(grouped):
        raw_rows = grouped[tid]
        raw_frames = [int(r["frame_idx"]) for r in raw_rows]
        for prev_f, next_f in zip(raw_frames, raw_frames[1:]):
            if next_f < prev_f:
                out_of_order_events.append({"track_id": tid, "prev_frame": prev_f, "next_frame": next_f})

        sorted_rows = sorted(raw_rows, key=lambda r: (int(r["frame_idx"]), float(r["timestamp_sec"])))
        frame_counts = Counter(int(r["frame_idx"]) for r in sorted_rows)
        for fr, cnt in sorted(frame_counts.items()):
            if cnt > 1:
                duplicate_frame_events.append({"track_id": tid, "frame_idx": fr, "count": cnt})

        classes_seen = sorted({str(r["class_name"]) for r in sorted_rows})
        if len(classes_seen) > 1:
            class_inconsistency_events.append({"track_id": tid, "classes_seen": classes_seen})

        frames: List[int] = []
        source_frames: List[int] = []
        timestamps: List[float] = []
        centers_xy: List[List[float]] = []
        vel_px_frame: List[Optional[List[float]]] = []
        vel_px_sec: List[Optional[List[float]]] = []
        steps: List[Dict] = []

        for row in sorted_rows:
            fr = int(row["frame_idx"])
            sfr = int(row.get("source_frame_idx", fr))
            ts = float(row["timestamp_sec"])
            cx = round(float(row["cx"]), 2)
            cy = round(float(row["cy"]), 2)

            vx_s: Optional[float] = None
            vy_s: Optional[float] = None
            if frames:
                df = fr - frames[-1]
                dt_sec = ts - timestamps[-1]
                px, py = centers_xy[-1]
                if df > 0:
                    vel_px_frame.append([round((cx - px) / df, 2), round((cy - py) / df, 2)])
                else:
                    vel_px_frame.append(None)

                if dt_sec > 0:
                    vx_s = round((cx - px) / dt_sec, 2)
                    vy_s = round((cy - py) / dt_sec, 2)
                    vel_px_sec.append([vx_s, vy_s])
                else:
                    vel_px_sec.append(None)

            frames.append(fr)
            source_frames.append(sfr)
            timestamps.append(ts)
            centers_xy.append([cx, cy])
            steps.append({
                "frame_idx": fr,
                "source_frame_idx": sfr,
                "timestamp_sec": round(ts, 3),
                "cx": cx,
                "cy": cy,
                "vx_px_per_sec": vx_s,
                "vy_px_per_sec": vy_s,
            })

        tracks_out.append({
            "track_id": tid,
            "class_name": sorted_rows[0]["class_name"],
            "classes_seen": classes_seen,
            "frames": frames,
            "source_frames": source_frames,
            "timestamps_sec": timestamps,
            "centers_xy": centers_xy,
            "velocities_px_per_frame": vel_px_frame,
            "velocities_px_per_sec": vel_px_sec,
            "steps": steps,
        })

    return {
        "total_records": len(records),
        "valid_tracked_records": len(records) - untracked_count,
        "untracked_detection_records": untracked_count,
        "unique_track_ids_in_clip": len(tracks_out),
        "velocity_coordinate_space": "2d_image_pixels_not_physical_world_meters",
        "validation_issues": {
            "out_of_order_events": out_of_order_events,
            "duplicate_frame_events": duplicate_frame_events,
            "class_inconsistency_events": class_inconsistency_events,
        },
        "tracks": tracks_out,
    }


def normalize_kitti_class_name(raw_name: str) -> Optional[str]:
    """Map raw detector label names cleanly to canonical KITTI Tracking class names."""
    return KITTI_CLASS_MAP.get(str(raw_name).strip().lower())


def export_tracking_outputs(
    records: Sequence[Dict],
    csv_path: Path,
    json_path: Path,
    kitti_txt_path: Optional[Path] = None,
) -> Dict:
    """Save CSV detection records, validated JSON trajectories, and optional KITTI 2D track txt."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx", "source_frame_idx", "timestamp_sec", "track_id", "class_id", "class_name",
        "conf", "x1", "y1", "x2", "y2", "cx", "cy",
        "mask_area_px", "mask_fill_ratio", "mask_bottom_x", "mask_bottom_y",
    ]
    normalized_rows = []
    for row in records:
        out_row = dict(row)
        out_row.setdefault("source_frame_idx", row["frame_idx"])
        out_row.setdefault("mask_area_px", 0)
        out_row.setdefault("mask_fill_ratio", "")
        out_row.setdefault("mask_bottom_x", "")
        out_row.setdefault("mask_bottom_y", "")
        normalized_rows.append({k: out_row[k] for k in fieldnames})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    summary = compute_track_trajectories(normalized_rows)

    if kitti_txt_path is not None:
        kitti_txt_path.parent.mkdir(parents=True, exist_ok=True)
        written_rows = 0
        skipped_classes = Counter()
        with open(kitti_txt_path, "w", encoding="utf-8") as kf:
            for r in normalized_rows:
                if int(r["track_id"]) < 0:
                    continue
                kitti_cls = normalize_kitti_class_name(r["class_name"])
                if kitti_cls is None:
                    skipped_classes[str(r["class_name"])] += 1
                    continue
                fr_out = int(r["source_frame_idx"])
                # 18 KITTI columns: 2D bbox + track_id only; 3D fields are TrackEval placeholders
                line = (
                    f"{fr_out} {int(r['track_id'])} {kitti_cls} "
                    f"-1 -1 -10 {float(r['x1']):.2f} {float(r['y1']):.2f} {float(r['x2']):.2f} {float(r['y2']):.2f} "
                    f"-1 -1 -1 -1000 -1000 -1000 -10 {float(r['conf']):.4f}\n"
                )
                kf.write(line)
                written_rows += 1
        summary["kitti_export"] = {
            "path": str(kitti_txt_path),
            "prediction_scope": "2d_bbox_and_track_id_only_3d_fields_are_placeholders",
            "written_rows": written_rows,
            "skipped_unmapped_classes": dict(skipped_classes),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def draw_tracking_frame(
    frame,
    frame_rows: Sequence[Dict],
    trail_history: Dict[int, List[Tuple[int, int]]],
    max_trail_len: int = 25,
    frame_masks: Optional[Dict[int, List[List[int]]]] = None,
):
    """Render instance segmentation masks, bounding boxes, track IDs, and trails onto a frame."""
    import cv2
    import numpy as np

    canvas = frame.copy()
    mask_overlay = frame.copy()
    has_masks = False
    palette = [
        (0, 220, 0), (255, 190, 0), (0, 140, 255),
        (200, 90, 255), (255, 255, 0), (0, 255, 255),
    ]
    for r in frame_rows:
        tid = int(r["track_id"])
        color = palette[tid % len(palette)] if tid >= 0 else (180, 180, 180)
        poly = r.get("polygon_xy") or (frame_masks.get(tid) if frame_masks and tid >= 0 else None)
        if poly and len(poly) >= 3:
            pts_poly = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_overlay, [pts_poly], color)
            has_masks = True

    if has_masks:
        canvas = cv2.addWeighted(mask_overlay, 0.35, canvas, 0.65, 0.0)

    for r in frame_rows:
        tid = int(r["track_id"])
        color = palette[tid % len(palette)] if tid >= 0 else (180, 180, 180)
        x1, y1, x2, y2 = int(round(r["x1"])), int(round(r["y1"])), int(round(r["x2"])), int(round(r["y2"]))
        cx, cy = int(round(r["cx"])), int(round(r["cy"]))

        poly = r.get("polygon_xy") or (frame_masks.get(tid) if frame_masks and tid >= 0 else None)
        if poly and len(poly) >= 3:
            pts_poly = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.circle(canvas, (cx, cy), 4, color, -1)
        cv2.putText(
            canvas,
            f"{r['class_name']} #{tid} {r['conf']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        if tid >= 0:
            pts = trail_history[tid]
            pts.append((cx, cy))
            if len(pts) > max_trail_len:
                del pts[:-max_trail_len]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2)
    return canvas


def _extract_instance_masks_sam(
    bgr,
    boxes_xyxy: Sequence[Sequence[float]],
    sam_model,
) -> List[Optional[Dict]]:
    """Run box-prompted SAM instance masking and extract contours + bottom contact points."""
    import cv2
    import numpy as np

    if sam_model is None or not boxes_xyxy:
        return [None] * len(boxes_xyxy)

    h, w = bgr.shape[:2]
    try:
        seg_res = sam_model(bgr, bboxes=[list(map(float, b)) for b in boxes_xyxy], verbose=False)[0]
    except Exception:
        return [None] * len(boxes_xyxy)

    if seg_res.masks is None or seg_res.masks.data is None:
        return [None] * len(boxes_xyxy)

    raw_masks = seg_res.masks.data.cpu().numpy()
    out: List[Optional[Dict]] = []
    for idx, box in enumerate(boxes_xyxy):
        if idx >= len(raw_masks):
            out.append(None)
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        bx1 = max(0, min(w - 1, int(math.floor(x1))))
        by1 = max(0, min(h - 1, int(math.floor(y1))))
        bx2 = max(bx1 + 1, min(w, int(math.ceil(x2))))
        by2 = max(by1 + 1, min(h, int(math.ceil(y2))))

        m = raw_masks[idx]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        bin_mask = np.zeros((h, w), dtype=np.uint8)
        bin_mask[by1:by2, bx1:bx2] = (m[by1:by2, bx1:bx2] > 0.5).astype(np.uint8)

        area_px = int(bin_mask.sum())
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        if area_px < 12:
            out.append(None)
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out.append(None)
            continue
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, max(1.5, 0.008 * peri), True)
        polygon_xy = [[int(pt[0][0]), int(pt[0][1])] for pt in approx[:64]]

        ys, xs = np.where(bin_mask > 0)
        y_98 = float(np.percentile(ys, 98))
        bottom_band = ys >= (y_98 - max(2.0, 0.12 * (y2 - y1)))
        x_bot = float(np.median(xs[bottom_band])) if np.any(bottom_band) else 0.5 * (x1 + x2)

        out.append(
            {
                "mask_area_px": area_px,
                "mask_fill_ratio": round(area_px / box_area, 4),
                "mask_bottom_x": round(x_bot, 2),
                "mask_bottom_y": round(y_98, 2),
                "polygon_xy": polygon_xy,
            }
        )
    return out


def inspect_image_directory(source: Path) -> Tuple[List[Tuple[int, Path]], List[Tuple[int, int]]]:
    """
    Collect and numerically sort image frames in a directory, returning
    ([(source_frame_idx, path), ...], [(prev_fr, curr_fr), ...] gaps).
    """
    raw_paths = [p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not raw_paths:
        raise FileNotFoundError(f"No image frames in {source}")
    indexed = sorted(
        ((extract_source_frame_idx(p, i), p) for i, p in enumerate(sorted(raw_paths, key=lambda x: x.name))),
        key=lambda item: (item[0], item[1].name),
    )
    frame_nums = [fr for fr, _ in indexed]
    if len(frame_nums) != len(set(frame_nums)):
        raise ValueError(f"Duplicate frame indices in directory {source}")
    gaps = [(a, b) for a, b in zip(frame_nums, frame_nums[1:]) if b != a + 1]
    return indexed, gaps


def run_traffic_tracking(
    weights: Path,
    source: Path,
    output_dir: Path,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int = 960,
    mask_model: Optional[str] = "mobile_sam.pt",
    fps_override: Optional[float] = None,
    max_frames: Optional[int] = None,
    kitti_seq_name: Optional[str] = None,
    require_contiguous: bool = True,
) -> Dict:
    """Run YOLO tracking + optional box-prompted SAM instance masking over images or video."""
    import cv2
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    sam = None
    mask_backend = "none"
    if mask_model and str(mask_model).lower() not in {"none", "off", "false"}:
        try:
            from ultralytics import SAM

            sam = SAM(str(mask_model))
            mask_backend = str(mask_model)
        except Exception as exc:
            print(f"[WARN] Could not initialize mask model '{mask_model}': {exc}", flush=True)
            sam = None

    indexed_paths: List[Tuple[int, Path]] = []
    frame_gaps: List[Tuple[int, int]] = []
    if source.is_dir():
        if fps_override is None or fps_override <= 0:
            raise ValueError("Explicit positive --fps is required for image directory sources (e.g., --fps 10 for KITTI).")
        indexed_paths, frame_gaps = inspect_image_directory(source)
        if require_contiguous and frame_gaps:
            raise ValueError(f"Non-contiguous frames in {source}: gaps at {frame_gaps[:5]}")
        first = cv2.imread(str(indexed_paths[0][1]))
        if first is None:
            raise RuntimeError(f"Cannot decode first frame: {indexed_paths[0][1]}")
        h, w = first.shape[:2]
        fps = float(fps_override)
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {source}")
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fps = float(fps_override) if fps_override is not None else raw_fps
        if fps <= 0:
            raise ValueError(f"Video {source} reported invalid FPS ({raw_fps}); pass positive --fps explicitly.")

    video_out = output_dir / "tracking_demo.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {video_out}")

    records: List[Dict] = []
    masks_by_frame: Dict[str, List[Dict]] = {}
    masked_instances_count = 0
    trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    unreadable_frames: List[str] = []
    clipped_box_count = 0
    processed = 0
    timestamps_seen: List[float] = []

    def handle_frame(idx: int, source_fr: int, ts_sec: float, bgr) -> None:
        nonlocal clipped_box_count, masked_instances_count
        if bgr.shape[:2] != (h, w):
            raise ValueError(f"Frame size mismatch at source_frame={source_fr}: expected {(h, w)}, got {bgr.shape[:2]}")
        res = model.track(bgr, imgsz=imgsz, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
        frame_rows: List[Dict] = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            tids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(boxes)
            clipped_boxes: List[Tuple[float, float, float, float]] = []
            for box in xyxys:
                rx1, ry1, rx2, ry2 = [float(v) for v in box]
                x1, y1, x2, y2, was_clipped = clip_xyxy_to_bounds(rx1, ry1, rx2, ry2, w, h)
                if was_clipped:
                    clipped_box_count += 1
                clipped_boxes.append((x1, y1, x2, y2))

            mask_infos = _extract_instance_masks_sam(bgr, clipped_boxes, sam) if sam is not None else [None] * len(clipped_boxes)
            frame_mask_entries: List[Dict] = []

            for (x1, y1, x2, y2), cf, cid, tid, minfo in zip(clipped_boxes, confs, clss, tids, mask_infos):
                row = {
                    "frame_idx": idx,
                    "source_frame_idx": source_fr,
                    "timestamp_sec": round(ts_sec, 3),
                    "track_id": int(tid),
                    "class_id": int(cid),
                    "class_name": model.names.get(int(cid), str(cid)),
                    "conf": round(float(cf), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "cx": round((x1 + x2) / 2.0, 2),
                    "cy": round((y1 + y2) / 2.0, 2),
                    "mask_area_px": minfo["mask_area_px"] if minfo else 0,
                    "mask_fill_ratio": minfo["mask_fill_ratio"] if minfo else "",
                    "mask_bottom_x": minfo["mask_bottom_x"] if minfo else "",
                    "mask_bottom_y": minfo["mask_bottom_y"] if minfo else "",
                }
                if minfo is not None:
                    row["polygon_xy"] = minfo["polygon_xy"]
                    masked_instances_count += 1
                    frame_mask_entries.append(
                        {
                            "track_id": int(tid),
                            "class_name": row["class_name"],
                            "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                            "mask_area_px": minfo["mask_area_px"],
                            "mask_fill_ratio": minfo["mask_fill_ratio"],
                            "mask_bottom_x": minfo["mask_bottom_x"],
                            "mask_bottom_y": minfo["mask_bottom_y"],
                            "polygon_xy": minfo["polygon_xy"],
                        }
                    )
                records.append(row)
                frame_rows.append(row)
            if frame_mask_entries:
                masks_by_frame[str(source_fr)] = frame_mask_entries
        writer.write(draw_tracking_frame(bgr, frame_rows, trails))
        timestamps_seen.append(ts_sec)

    cap = None
    try:
        if source.is_dir():
            base_sfr = indexed_paths[0][0]
            for idx, (sfr, p) in enumerate(indexed_paths):
                if max_frames is not None and idx >= max_frames:
                    break
                bgr = cv2.imread(str(p))
                if bgr is None:
                    unreadable_frames.append(str(p.name))
                    continue
                handle_frame(idx, sfr, (sfr - base_sfr) / fps, bgr)
                processed += 1
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to reopen {source}")
            idx = 0
            while cap.isOpened():
                if max_frames is not None and idx >= max_frames:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                handle_frame(idx, idx, idx / fps, bgr)
                idx += 1
                processed += 1
    finally:
        writer.release()
        if cap is not None:
            cap.release()

    masks_json_path = output_dir / "trajectories_masks.json"
    with open(masks_json_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "mask_backend": mask_backend,
                "masked_instances_count": masked_instances_count,
                "masks_by_frame": masks_by_frame,
            },
            mf,
        )

    kitti_txt = output_dir / "kitti_preds" / f"{kitti_seq_name}.txt" if kitti_seq_name else None
    summary = export_tracking_outputs(
        records,
        csv_path=output_dir / "trajectories.csv",
        json_path=output_dir / "trajectories.json",
        kitti_txt_path=kitti_txt,
    )
    summary["frames_processed"] = processed
    summary["unreadable_frames"] = unreadable_frames
    summary["clipped_box_count"] = clipped_box_count
    summary["frame_gaps"] = frame_gaps
    summary["fps"] = fps
    summary["imgsz"] = imgsz
    summary["mask_backend"] = mask_backend
    summary["masked_instances_count"] = masked_instances_count
    summary["masks_json_path"] = str(masks_json_path)
    summary["video_duration_sec"] = round(processed / fps, 3)
    summary["observation_span_sec"] = round(max(timestamps_seen) - min(timestamps_seen), 3) if len(timestamps_seen) >= 2 else 0.0
    summary["video_path"] = str(video_out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Total Care Traffic YOLO Tracking & Instance Masking")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Image sequence directory or video file")
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/track"))
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--mask-model", type=str, default="mobile_sam.pt", help="SAM model weights or 'none'")
    parser.add_argument("--fps", type=float, default=None, help="Required for image directory sources (e.g. 10 for KITTI)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--kitti-seq", type=str, default=None, help="If set, export 18-col KITTI 2D prediction txt")
    parser.add_argument("--allow-noncontiguous", action="store_true", help="Allow gaps between frame numbers without error")
    args = parser.parse_args()

    out = run_traffic_tracking(
        weights=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        mask_model=args.mask_model,
        fps_override=args.fps,
        max_frames=args.max_frames,
        kitti_seq_name=args.kitti_seq,
        require_contiguous=not args.allow_noncontiguous,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
