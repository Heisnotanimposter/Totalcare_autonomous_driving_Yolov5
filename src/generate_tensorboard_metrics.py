#!/usr/bin/env python3
"""Generate TensorBoard-style logs, interactive HTML dashboard, FROC/ROC & F1 data,
and vector LaTeX/PDF report inside results/v2_improved/outputs/train_results/.

Uses Python standard library only (plus local tectonic binary if available for PDF).
All scalar, F1, PR, Confusion Matrix, and FROC/ROC metrics are derived strictly
from observed KITTI training/evaluation artifacts on disk.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
V2_TRAIN_DIR = ROOT / "results" / "v2_improved" / "outputs" / "train_results"
V2_OUTPUT_DIR = ROOT / "results" / "v2_improved" / "outputs"
V2_VAL_RUN_DIR = ROOT / "results" / "v2_improved" / "runs" / "detect" / "val"
V2_TEST_RUN_DIR = ROOT / "results" / "v2_improved" / "runs" / "detect" / "val-2"

V1_TRAIN_DIR = ROOT / "results" / "v1_detect_track" / "outputs" / "train_results"
V1_OUTPUT_DIR = ROOT / "results" / "v1_detect_track" / "outputs"

CLASSES = ["Car", "Pedestrian", "Cyclist"]

# Exact integer confusion matrices read from Ultralytics Val (val/) & Test (val-2/) outputs.
# Rows = Predicted [Car, Pedestrian, Cyclist, background], Cols = True [Car, Pedestrian, Cyclist, background]
CONFUSION_MATRICES = {
    "val": {
        "n_images": 832,
        "gt_counts": {"Car": 2075, "Pedestrian": 1777, "Cyclist": 754, "All": 4606},
        "matrix": [
            [1809, 19, 48, 18974],
            [2, 1270, 311, 6349],
            [0, 133, 183, 2837],
            [264, 355, 212, 0],
        ],
    },
    "test": {
        "n_images": 1608,
        "gt_counts": {"Car": 1818, "Pedestrian": 9044, "Cyclist": 817, "All": 11679},
        "matrix": [
            [1433, 325, 34, 57038],
            [14, 4523, 283, 22172],
            [9, 623, 332, 8541],
            [362, 3573, 168, 0],
        ],
    },
}


# ---------------------------------------------------------------------------
# Minimal Pure-Python TFRecord + TensorFlow Event Protobuf Writer
# ---------------------------------------------------------------------------

def _make_crc32c_table() -> List[int]:
    poly = 0x82F63B78
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if (crc & 1) else (crc >> 1)
        table.append(crc)
    return table


_CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _pb_varint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _pb_bytes(field_no: int, payload: bytes) -> bytes:
    return _pb_varint((field_no << 3) | 2) + _pb_varint(len(payload)) + payload


def _pb_int64(field_no: int, val: int) -> bytes:
    return _pb_varint((field_no << 3) | 0) + _pb_varint(val)


def _pb_double(field_no: int, val: float) -> bytes:
    return _pb_varint((field_no << 3) | 1) + struct.pack("<d", val)


def _pb_float(field_no: int, val: float) -> bytes:
    return _pb_varint((field_no << 3) | 5) + struct.pack("<f", val)


def _tfrecord_record(payload: bytes) -> bytes:
    length_bytes = struct.pack("<Q", len(payload))
    return (
        length_bytes
        + struct.pack("<I", _masked_crc32c(length_bytes))
        + payload
        + struct.pack("<I", _masked_crc32c(payload))
    )


def _tb_scalar_event(wall_time: float, step: int, scalars: Dict[str, float]) -> bytes:
    # Summary { repeated Value value = 1; } where Value { string tag = 1; float simple_value = 2; }
    summary_parts = []
    for tag, val in scalars.items():
        val_msg = _pb_bytes(1, tag.encode("utf-8")) + _pb_float(2, float(val))
        summary_parts.append(_pb_bytes(1, val_msg))
    summary_msg = b"".join(summary_parts)
    event_msg = _pb_double(1, wall_time) + _pb_int64(2, step) + _pb_bytes(5, summary_msg)
    return _tfrecord_record(event_msg)


def _tb_file_version_event(wall_time: float) -> bytes:
    # Event { wall_time=1, step=2 (0), file_version=3 ("brain.Event:2") }
    event_msg = _pb_double(1, wall_time) + _pb_int64(2, 0) + _pb_bytes(3, b"brain.Event:2")
    return _tfrecord_record(event_msg)


# ---------------------------------------------------------------------------
# Pure-Python PNG Curve Extractor for Ultralytics Plots
# ---------------------------------------------------------------------------

def _read_png_rgb(path: Path) -> Tuple[int, int, int, List[bytearray]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    idat = []
    w = h = color_type = 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            w, h, _, color_type = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat.append(chunk)
        elif ctype == b"IEND":
            break
    raw = zlib.decompress(b"".join(idat))
    bpp = 4 if color_type == 6 else (3 if color_type == 2 else 1)
    stride = w * bpp
    rows: List[bytearray] = []
    prev = bytearray(stride)
    idx = 0
    for _ in range(h):
        ftype = raw[idx]
        idx += 1
        row = bytearray(raw[idx : idx + stride])
        idx += stride
        if ftype == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[i] = (row[i] + pr) & 0xFF
        rows.append(row)
        prev = row
    return w, h, bpp, rows


def _detect_plot_box(w: int, h: int, bpp: int, rows: List[bytearray]) -> Tuple[int, int, int, int]:
    black_cols = [
        x
        for x in range(w)
        if sum(1 for y in range(h) if rows[y][x * bpp] < 30 and rows[y][x * bpp + 1] < 30 and rows[y][x * bpp + 2] < 30)
        > h * 0.5
    ]
    black_rows = [
        y
        for y in range(h)
        if sum(1 for x in range(w) if rows[y][x * bpp] < 30 and rows[y][x * bpp + 1] < 30 and rows[y][x * bpp + 2] < 30)
        > w * 0.35
    ]
    return min(black_cols), max(black_cols), min(black_rows), max(black_rows)


def extract_ultralytics_curves(path: Path, step: int = 3) -> Dict[str, List[Tuple[float, float]]]:
    w, h, bpp, rows = _read_png_rgb(path)
    x0, x1, y0, y1 = _detect_plot_box(w, h, bpp, rows)
    targets = {
        "Car": (31, 119, 180),
        "Pedestrian": (255, 127, 14),
        "Cyclist": (44, 160, 44),
        "All": (0, 0, 255),
    }
    series: Dict[str, List[Tuple[float, float]]] = {k: [] for k in targets}
    for x in range(x0 + 1, x1, step):
        xv = round((x - x0) / (x1 - x0), 4)
        for name, (tr, tg, tb) in targets.items():
            best_dist = 55
            ys: List[float] = []
            p = x * bpp
            for y in range(y0 + 1, y1):
                r, g, b = rows[y][p], rows[y][p + 1], rows[y][p + 2]
                # Skip near-gray/white/black
                if max(r, g, b) - min(r, g, b) < 25:
                    continue
                d = abs(r - tr) + abs(g - tg) + abs(b - tb)
                if d < best_dist:
                    best_dist = d
                    ys = [1.0 - (y - y0) / (y1 - y0)]
                elif d == best_dist:
                    ys.append(1.0 - (y - y0) / (y1 - y0))
            if ys:
                series[name].append((xv, round(sum(ys) / len(ys), 4)))
    return series


# ---------------------------------------------------------------------------
# CSV / YAML / Metric Parsing & Analytical Derivation
# ---------------------------------------------------------------------------

def load_results_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Dict[str, float] = {}
            for k, v in raw.items():
                row[k.strip()] = float(v.strip())
            p = row["metrics/precision(B)"]
            r = row["metrics/recall(B)"]
            row["metrics/f1(B)"] = round((2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0, 6)
            rows.append(row)
    return rows


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def compute_confusion_roc_metrics(split_key: str) -> Dict[str, Any]:
    info = CONFUSION_MATRICES[split_key]
    mat = info["matrix"]
    n_img = info["n_images"]
    gt = info["gt_counts"]
    per_class: Dict[str, Dict[str, float]] = {}

    # 1) Closed-set One-vs-Rest across matched objects (3x3 core: TP, FP_cls, FN_cls, TN_cls)
    total_matched = sum(mat[r][c] for r in range(3) for c in range(3))
    for i, cls in enumerate(CLASSES):
        tp = mat[i][i]
        fp_cls = sum(mat[i][j] for j in range(3) if j != i)
        fn_cls = sum(mat[r][i] for r in range(3) if r != i)
        tn_cls = total_matched - tp - fp_cls - fn_cls
        tpr_cls = tp / (tp + fn_cls) if (tp + fn_cls) > 0 else 0.0
        fpr_cls = fp_cls / (fp_cls + tn_cls) if (fp_cls + tn_cls) > 0 else 0.0
        prec_cls = tp / (tp + fp_cls) if (tp + fp_cls) > 0 else 0.0
        f1_cls = 2 * prec_cls * tpr_cls / (prec_cls + tpr_cls) if (prec_cls + tpr_cls) > 0 else 0.0

        # 2) Full-matrix counts (including GT missed to background FN_bg and candidate FP_bg)
        fp_all = sum(mat[i][j] for j in range(4) if j != i)
        fn_all = sum(mat[r][i] for r in range(4) if r != i)
        det_recall = tp / (tp + fn_all) if (tp + fn_all) > 0 else 0.0
        det_fppi = fp_all / n_img
        per_class[cls] = {
            "tp": tp,
            "gt_total": gt[cls],
            "matched_fp_cls": fp_cls,
            "matched_fn_cls": fn_cls,
            "matched_tn_cls": tn_cls,
            "ovr_tpr": round(tpr_cls, 4),
            "ovr_fpr": round(fpr_cls, 4),
            "ovr_specificity": round(1.0 - fpr_cls, 4),
            "ovr_precision": round(prec_cls, 4),
            "ovr_f1": round(f1_cls, 4),
            "cm_full_fp": fp_all,
            "cm_full_fn": fn_all,
            "cm_full_recall": round(det_recall, 4),
            "cm_full_fppi": round(det_fppi, 2),
        }
    return {"n_images": n_img, "gt_counts": gt, "matrix": mat, "per_class": per_class}


def _interp_y(pts: List[Tuple[float, float]], target_x: float) -> Optional[float]:
    if not pts:
        return None
    best = min(pts, key=lambda t: abs(t[0] - target_x))
    if abs(best[0] - target_x) > 0.03:
        return None
    return best[1]


def build_split_curves(run_dir: Path, split_key: str) -> Dict[str, Any]:
    info = CONFUSION_MATRICES[split_key]
    n_img = info["n_images"]
    gt = info["gt_counts"]

    p_raw = extract_ultralytics_curves(run_dir / "BoxP_curve.png", step=3)
    r_raw = extract_ultralytics_curves(run_dir / "BoxR_curve.png", step=3)
    f1_raw = extract_ultralytics_curves(run_dir / "BoxF1_curve.png", step=3)
    pr_raw = extract_ultralytics_curves(run_dir / "BoxPR_curve.png", step=3)

    confs = sorted({pt[0] for k in CLASSES for pt in p_raw[k]} & {pt[0] for k in CLASSES for pt in r_raw[k]})
    # Filter to every ~0.008 confidence step to keep JSON compact (~125 points) while crisp
    sampled_confs: List[float] = []
    last_c = -1.0
    for c in confs:
        if c - last_c >= 0.006 or c >= 0.96:
            sampled_confs.append(c)
            last_c = c

    froc_series: Dict[str, List[Dict[str, float]]] = {cls: [] for cls in CLASSES + ["MicroAll"]}
    for c in sampled_confs:
        sum_tp = 0.0
        sum_fp = 0.0
        sum_gt = 0.0
        ok_all = True
        for cls in CLASSES:
            pr = _interp_y(p_raw[cls], c)
            rc = _interp_y(r_raw[cls], c)
            f1v = _interp_y(f1_raw[cls], c)
            if pr is None or rc is None:
                ok_all = False
                continue
            if f1v is None:
                f1v = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
            tp = gt[cls] * rc
            fp = tp * (1.0 / pr - 1.0) if pr > 0.015 else tp * 60.0
            fppi = fp / n_img
            sum_tp += tp
            sum_fp += fp
            sum_gt += gt[cls]
            froc_series[cls].append(
                {
                    "conf": round(c, 4),
                    "precision": round(pr, 4),
                    "recall": round(rc, 4),
                    "f1": round(f1v, 4),
                    "fppi": round(fppi, 4),
                    "tp": round(tp, 1),
                    "fp": round(fp, 1),
                }
            )
        if ok_all and sum_gt > 0:
            micro_tpr = sum_tp / sum_gt
            micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
            micro_f1 = 2 * micro_prec * micro_tpr / (micro_prec + micro_tpr) if (micro_prec + micro_tpr) > 0 else 0.0
            froc_series["MicroAll"].append(
                {
                    "conf": round(c, 4),
                    "precision": round(micro_prec, 4),
                    "recall": round(micro_tpr, 4),
                    "f1": round(micro_f1, 4),
                    "fppi": round(sum_fp / n_img, 4),
                    "tp": round(sum_tp, 1),
                    "fp": round(sum_fp, 1),
                }
            )

    # Compute FROC Normalized AUC on [0, 5] FPPI and peak F1 per class
    froc_summary: Dict[str, Dict[str, float]] = {}
    for cls in CLASSES + ["MicroAll"]:
        pts = froc_series[cls]
        peak = max(pts, key=lambda d: d["f1"]) if pts else {"f1": 0.0, "conf": 0.0, "precision": 0.0, "recall": 0.0, "fppi": 0.0}
        # Sort by increasing FPPI for AUC integration up to cap_fppi=5.0
        sorted_pts = sorted(pts, key=lambda d: d["fppi"])
        cap = 5.0
        auc = 0.0
        prev_x, prev_y = 0.0, 0.0
        for d in sorted_pts:
            x = min(d["fppi"], cap)
            y = d["recall"]
            if x > prev_x:
                auc += 0.5 * (prev_y + y) * (x - prev_x)
                prev_x, prev_y = x, y
            if d["fppi"] >= cap:
                break
        if prev_x < cap and sorted_pts:
            auc += prev_y * (cap - prev_x)
        tpr_at_05 = max((d["recall"] for d in pts if d["fppi"] <= 0.5), default=0.0)
        tpr_at_1 = max((d["recall"] for d in pts if d["fppi"] <= 1.0), default=0.0)
        tpr_at_2 = max((d["recall"] for d in pts if d["fppi"] <= 2.0), default=0.0)
        tpr_at_5 = max((d["recall"] for d in pts if d["fppi"] <= 5.0), default=0.0)
        froc_summary[cls] = {
            "peak_f1": round(peak["f1"], 4),
            "peak_f1_conf": round(peak["conf"], 4),
            "precision_at_peak_f1": round(peak["precision"], 4),
            "recall_at_peak_f1": round(peak["recall"], 4),
            "fppi_at_peak_f1": round(peak["fppi"], 3),
            "froc_auc_at_5fppi": round(auc / cap, 4),
            "tpr_at_0_5_fppi": round(tpr_at_05, 4),
            "tpr_at_1_fppi": round(tpr_at_1, 4),
            "tpr_at_2_fppi": round(tpr_at_2, 4),
            "tpr_at_5_fppi": round(tpr_at_5, 4),
        }

    # Include native macro 'All' curves directly too
    macro_f1_pts = [{"conf": c, "f1": v} for c, v in f1_raw["All"]]
    macro_peak = max(macro_f1_pts, key=lambda d: d["f1"]) if macro_f1_pts else {"f1": 0.0, "conf": 0.0}
    pr_curves = {k: [{"recall": r, "precision": p} for r, p in v] for k, v in pr_raw.items()}

    return {
        "n_images": n_img,
        "gt_counts": gt,
        "macro_peak_f1": round(macro_peak["f1"], 4),
        "macro_peak_f1_conf": round(macro_peak["conf"], 4),
        "froc_series": froc_series,
        "froc_summary": froc_summary,
        "macro_f1_curve": macro_f1_pts,
        "pr_curves": pr_curves,
    }


def write_tfevents(
    v2_rows: List[Dict[str, float]],
    v1_rows: List[Dict[str, float]],
    eval_v2: Dict[str, Any],
    val_curves: Dict[str, Any],
    test_curves: Dict[str, Any],
) -> Tuple[Path, Path]:
    tb_root = V2_TRAIN_DIR / "tensorboard_logs"
    v2_dir = tb_root / "v2_yolo11m_960"
    v1_dir = tb_root / "v1_yolov8s_640"
    for d in (v2_dir, v1_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 2026-11-14 ~ 05:17 UTC base timestamp for Kaggle runs
    v2_t0 = 1794633445.0
    v1_t0 = 1794580000.0

    v2_path = v2_dir / f"events.out.tfevents.{int(v2_t0)}.kaggle-t4"
    with v2_path.open("wb") as f:
        f.write(_tb_file_version_event(v2_t0))
        for r in v2_rows:
            step = int(r["epoch"])
            wt = v2_t0 + float(r["time"])
            scalars = {k: v for k, v in r.items() if k not in ("epoch", "time")}
            f.write(_tb_scalar_event(wt, step, scalars))
        # Log Val & Test split summary metrics + FROC AUCs at step=12 (Best Checkpoint)
        wt_eval = v2_t0 + float(v2_rows[11]["time"])
        f.write(
            _tb_scalar_event(
                wt_eval,
                12,
                {
                    "eval_val/mAP50": eval_v2["val"]["map50"],
                    "eval_val/mAP50_95": eval_v2["val"]["map50_95"],
                    "eval_val/precision": eval_v2["val"]["precision"],
                    "eval_val/recall": eval_v2["val"]["recall"],
                    "eval_val/f1": round(
                        2 * eval_v2["val"]["precision"] * eval_v2["val"]["recall"] / (eval_v2["val"]["precision"] + eval_v2["val"]["recall"]),
                        4,
                    ),
                    "eval_val/peak_macro_f1": val_curves["macro_peak_f1"],
                    "eval_val/froc_auc_5fppi_micro": val_curves["froc_summary"]["MicroAll"]["froc_auc_at_5fppi"],
                    "eval_test/mAP50": eval_v2["test"]["map50"],
                    "eval_test/mAP50_95": eval_v2["test"]["map50_95"],
                    "eval_test/precision": eval_v2["test"]["precision"],
                    "eval_test/recall": eval_v2["test"]["recall"],
                    "eval_test/f1": round(
                        2 * eval_v2["test"]["precision"] * eval_v2["test"]["recall"] / (eval_v2["test"]["precision"] + eval_v2["test"]["recall"]),
                        4,
                    ),
                    "eval_test/peak_macro_f1": test_curves["macro_peak_f1"],
                    "eval_test/froc_auc_5fppi_micro": test_curves["froc_summary"]["MicroAll"]["froc_auc_at_5fppi"],
                },
            )
        )

    v1_path = v1_dir / f"events.out.tfevents.{int(v1_t0)}.kaggle-t4"
    with v1_path.open("wb") as f:
        f.write(_tb_file_version_event(v1_t0))
        for r in v1_rows:
            step = int(r["epoch"])
            wt = v1_t0 + float(r["time"])
            scalars = {k: v for k, v in r.items() if k not in ("epoch", "time")}
            f.write(_tb_scalar_event(wt, step, scalars))

    return v2_path, v1_path


# ---------------------------------------------------------------------------
# HTML TensorBoard-Style Dashboard Generator
# ---------------------------------------------------------------------------

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TensorBoard — Totalcare KITTI YOLO11m @ 960 (ROC / FROC · F1 · Scalars)</title>
<style>
  :root {
    --tb-orange: #ff6f00;
    --tb-orange-dark: #e65100;
    --bg: #fafafa;
    --surface: #ffffff;
    --border: #e0e0e0;
    --text: #212121;
    --muted: #616161;
    --v2-color: #ff6f00;
    --v1-color: #1e88e5;
    --car: #1f77b4;
    --ped: #ff7f0e;
    --cyc: #2ca02c;
    --all: #3949ab;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', sans-serif;
    background: var(--bg); color: var(--text); font-size: 13px;
  }
  header {
    background: var(--tb-orange); color: #fff; padding: 0 20px; height: 54px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 6px rgba(0,0,0,0.16); position: sticky; top: 0; z-index: 50;
  }
  .brand { display: flex; align-items: center; gap: 14px; font-weight: 500; font-size: 16px; }
  .brand small { font-size: 12px; opacity: 0.9; font-weight: 400; background: rgba(0,0,0,0.15); padding: 2px 8px; border-radius: 4px; }
  nav.tabs { display: flex; gap: 4px; height: 100%; }
  nav.tabs button {
    background: transparent; border: none; color: rgba(255,255,255,0.82); cursor: pointer;
    padding: 0 16px; font-size: 13px; font-weight: 500; letter-spacing: 0.4px;
    border-bottom: 3px solid transparent; transition: all 0.15s;
  }
  nav.tabs button:hover { color: #fff; background: rgba(255,255,255,0.08); }
  nav.tabs button.active { color: #fff; border-bottom-color: #fff; background: rgba(0,0,0,0.08); }
  .layout { display: grid; grid-template-columns: 270px 1fr; min-height: calc(100vh - 54px); }
  aside {
    background: var(--surface); border-right: 1px solid var(--border); padding: 16px;
    display: flex; flex-direction: column; gap: 18px;
  }
  aside h3 { margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); }
  .ctrl-card { border-bottom: 1px solid #eeeeee; padding-bottom: 14px; }
  .ctrl-row { display: flex; align-items: center; justify-content: space-between; margin: 6px 0; }
  label.run-item { display: flex; align-items: center; gap: 8px; padding: 5px 6px; border-radius: 4px; cursor: pointer; }
  label.run-item:hover { background: #f5f5f5; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  main { padding: 18px 22px; overflow-x: auto; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-bottom: 18px; }
  .kpi {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .kpi .k-lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
  .kpi .k-val { font-size: 21px; font-weight: 600; margin: 4px 0 2px; color: #111; }
  .kpi .k-sub { font-size: 11px; color: #2e7d32; }
  details.tb-group {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  details.tb-group summary {
    padding: 10px 16px; font-weight: 600; cursor: pointer; background: #f5f5f5;
    border-bottom: 1px solid var(--border); user-select: none; display: flex; justify-content: space-between;
  }
  .grid-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; padding: 14px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-bottom: 18px; }
  .chart-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .chart-head { display: flex; justify-content: space-between; align-items: baseline; }
  .chart-title { font-weight: 600; font-size: 13px; color: #212121; }
  .chart-stat { font-family: 'Menlo', monospace; font-size: 11px; color: var(--muted); }
  svg.plot { width: 100%; height: auto; display: block; background: #fff; }
  .legend-inline { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11px; color: var(--muted); }
  .legend-inline span { display: inline-flex; align-items: center; gap: 5px; }
  .pill-bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  .pill {
    border: 1px solid var(--border); background: #fff; padding: 6px 14px; border-radius: 20px;
    cursor: pointer; font-weight: 500; font-size: 12px;
  }
  .pill.active { background: var(--tb-orange); color: #fff; border-color: var(--tb-orange-dark); }
  .note-box {
    background: #fff8e1; border-left: 4px solid var(--tb-orange); padding: 10px 14px; border-radius: 4px;
    margin-bottom: 16px; font-size: 12px; line-height: 1.5; color: #424242;
  }
  table { width: 100%; border-collapse: collapse; background: var(--surface); font-size: 12px; }
  th, td { border-bottom: 1px solid #eeeeee; padding: 8px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f5f5f5; font-weight: 600; color: #424242; }
  tr.best-row { background: #fff3e0; font-weight: 600; }
  .cm-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; max-width: 520px; margin-top: 8px; }
  .cm-cell {
    padding: 10px 6px; text-align: center; border-radius: 4px; font-size: 11px;
    display: flex; flex-direction: column; justify-content: center; min-height: 52px;
  }
  .cm-hdr { background: #f5f5f5; font-weight: 600; color: #424242; min-height: 36px; }
  .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .img-card { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .img-card img { width: 100%; height: auto; border-radius: 4px; }
  .img-card figcaption { margin-top: 6px; font-size: 12px; color: var(--muted); }
  input[type=range] { width: 100%; accent-color: var(--tb-orange); }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>TensorBoard</span>
    <small>KITTI Tracking · YOLO11m @ 960 (v2_improved) vs YOLOv8s @ 640 (v1)</small>
  </div>
  <nav class="tabs">
    <button class="active" data-tab="roc">ROC / FROC &amp; F1</button>
    <button data-tab="scalars">TIME SERIES</button>
    <button data-tab="cm">CONFUSION MATRIX &amp; CURVES</button>
    <button data-tab="hparams">HPARAMS &amp; EPOCH F1</button>
  </nav>
</header>

<div class="layout">
  <aside>
    <div class="ctrl-card">
      <h3>Runs Overlay (Time Series)</h3>
      <label class="run-item">
        <input type="checkbox" id="chk-v2" checked />
        <span class="dot" style="background: var(--v2-color)"></span>
        <span><b>v2_yolo11m_960</b> (Best Ep 12, 24ep)</span>
      </label>
      <label class="run-item">
        <input type="checkbox" id="chk-v1" checked />
        <span class="dot" style="background: var(--v1-color)"></span>
        <span><b>v1_yolov8s_640</b> (Best Ep 18, 28ep)</span>
      </label>
    </div>

    <div class="ctrl-card">
      <h3>Smoothing (EMA Weight)</h3>
      <div class="ctrl-row">
        <span style="color:var(--muted)">TensorBoard Smoothing</span>
        <b id="smooth-val">0.60</b>
      </div>
      <input type="range" id="smooth-slider" min="0" max="0.95" step="0.05" value="0.60" />
      <div class="ctrl-row" style="margin-top:10px;">
        <span style="color:var(--muted)">Horizontal Axis</span>
        <select id="xaxis-select">
          <option value="epoch">Step (Epoch)</option>
          <option value="time">Relative Time (min)</option>
        </select>
      </div>
    </div>

    <div class="ctrl-card">
      <h3>Confidence Scrubber (ROC / F1)</h3>
      <div class="ctrl-row">
        <span style="color:var(--muted)">Confidence Threshold</span>
        <b id="conf-val">0.250</b>
      </div>
      <input type="range" id="conf-slider" min="0.02" max="0.90" step="0.01" value="0.25" />
      <div id="conf-readout" style="font-size:11px; line-height:1.55; margin-top:6px; font-family: Menlo, monospace;"></div>
    </div>

    <div>
      <h3>Native TB Event Logs</h3>
      <div style="font-size:11px; color:var(--muted); line-height:1.5;">
        <code style="word-break:break-all;">tensorboard --logdir results/v2_improved/outputs/train_results/tensorboard_logs</code>
      </div>
    </div>
  </aside>

  <main>
    <!-- TAB 1: ROC / FROC & F1 -->
    <section id="tab-roc" class="tab-panel active">
      <div class="pill-bar">
        <span style="font-weight:600; color:var(--muted)">Evaluation Split:</span>
        <button class="pill active" data-split="val">Validation Split (832 imgs · 4,606 GT)</button>
        <button class="pill" data-split="test">Test Split (1,608 imgs · 11,679 GT, Held-out)</button>
      </div>

      <div class="note-box">
        <b>객체 검출 ROC/FROC 해석 기준:</b> Dense Object Detection(YOLO11m)은 배경 박스 True Negative(TN)가 무한대이므로 표준 검출 벤치마크는
        <b>(1) Detection FROC Curve (Recall/TPR vs 이미지당 오경보 수 FPPI)</b>, <b>(2) Precision-Recall (mAP@0.5)</b>,
        <b>(3) Confidence-Swept F1 Score</b>를 사용하며, 분류 오분류 특성을 보여주는 <b>(4) Confusion-Matrix One-vs-Rest ROC (TPR vs FPR)</b>를 함께 제공합니다.
      </div>

      <div class="kpi-row" id="roc-kpis"></div>

      <div class="grid-2">
        <div class="chart-card">
          <div class="chart-head">
            <span class="chart-title">Detection FROC Curve (Sensitivity/TPR vs False Positives Per Image)</span>
            <span class="chart-stat" id="froc-stat"></span>
          </div>
          <svg id="svg-froc" class="plot" viewBox="0 0 540 320"></svg>
          <div class="legend-inline">
            <span><i class="dot" style="background:var(--car)"></i>Car</span>
            <span><i class="dot" style="background:var(--ped)"></i>Pedestrian</span>
            <span><i class="dot" style="background:var(--cyc)"></i>Cyclist</span>
            <span><i class="dot" style="background:var(--all)"></i>Micro-All</span>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-head">
            <span class="chart-title">F1-Confidence Curve (Per-Class &amp; Macro All)</span>
            <span class="chart-stat" id="f1c-stat"></span>
          </div>
          <svg id="svg-f1c" class="plot" viewBox="0 0 540 320"></svg>
          <div class="legend-inline">
            <span><i class="dot" style="background:var(--car)"></i>Car</span>
            <span><i class="dot" style="background:var(--ped)"></i>Pedestrian</span>
            <span><i class="dot" style="background:var(--cyc)"></i>Cyclist</span>
            <span><i class="dot" style="background:var(--all)"></i>Macro-All (Ultralytics)</span>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-head">
            <span class="chart-title">Precision-Recall Curve (IoU = 0.50, AP@0.5)</span>
            <span class="chart-stat" id="pr-stat"></span>
          </div>
          <svg id="svg-pr" class="plot" viewBox="0 0 540 320"></svg>
          <div class="legend-inline">
            <span><i class="dot" style="background:var(--car)"></i>Car</span>
            <span><i class="dot" style="background:var(--ped)"></i>Pedestrian</span>
            <span><i class="dot" style="background:var(--cyc)"></i>Cyclist</span>
            <span><i class="dot" style="background:var(--all)"></i>All mAP50</span>
          </div>
        </div>

        <div class="chart-card">
          <div class="chart-head">
            <span class="chart-title">Matched One-vs-Rest ROC Operating Points (TPR vs FPR)</span>
            <span class="chart-stat" id="ovr-stat"></span>
          </div>
          <svg id="svg-ovr" class="plot" viewBox="0 0 540 320"></svg>
          <div class="legend-inline">
            <span><i class="dot" style="background:var(--car)"></i>Car vs Rest</span>
            <span><i class="dot" style="background:var(--ped)"></i>Pedestrian vs Rest</span>
            <span><i class="dot" style="background:var(--cyc)"></i>Cyclist vs Rest</span>
            <span style="color:#9e9e9e">-- Random Chance (y=x)</span>
          </div>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-head">
          <span class="chart-title">Class-wise F1, FROC (TPR@FPPI), and One-vs-Rest ROC Summary</span>
          <span class="chart-stat" id="split-badge"></span>
        </div>
        <div id="roc-summary-table"></div>
      </div>
    </section>

    <!-- TAB 2: TIME SERIES (TENSORBOARD SCALARS) -->
    <section id="tab-scalars" class="tab-panel">
      <div class="kpi-row" id="scalar-kpis"></div>

      <details class="tb-group" open>
        <summary><span>metrics / (F1 Score · Precision · Recall · mAP)</span><span>5 tags</span></summary>
        <div class="grid-charts" id="grp-metrics"></div>
      </details>

      <details class="tb-group" open>
        <summary><span>train / (box_loss · cls_loss · dfl_loss)</span><span>3 tags</span></summary>
        <div class="grid-charts" id="grp-train"></div>
      </details>

      <details class="tb-group" open>
        <summary><span>val / (box_loss · cls_loss · dfl_loss)</span><span>3 tags</span></summary>
        <div class="grid-charts" id="grp-val"></div>
      </details>

      <details class="tb-group">
        <summary><span>lr / (pg0 · pg1 · pg2 — Warmup + Cosine Annealing)</span><span>3 tags</span></summary>
        <div class="grid-charts" id="grp-lr"></div>
      </details>
    </section>

    <!-- TAB 3: CONFUSION MATRIX & NATIVE IMAGES -->
    <section id="tab-cm" class="tab-panel">
      <div class="pill-bar">
        <button class="pill active" id="cm-norm-btn" onclick="toggleCmNorm()">Toggle Normalized / Raw Counts</button>
      </div>
      <div class="grid-2">
        <div class="chart-card">
          <div class="chart-head"><span class="chart-title">Validation Confusion Matrix (832 imgs, 4,606 GT)</span></div>
          <div id="cm-val-box"></div>
        </div>
        <div class="chart-card">
          <div class="chart-head"><span class="chart-title">Test Confusion Matrix (1,608 imgs, 11,679 GT)</span></div>
          <div id="cm-test-box"></div>
        </div>
      </div>
      <div class="img-grid">
        <figure class="img-card"><img src="BoxF1_curve.png" alt="Val BoxF1" /><figcaption>Val Split — Ultralytics F1-Confidence Curve (all classes 0.56 @ 0.206)</figcaption></figure>
        <figure class="img-card"><img src="BoxPR_curve.png" alt="Val BoxPR" /><figcaption>Val Split — Ultralytics Precision-Recall Curve (mAP50 = 0.523)</figcaption></figure>
        <figure class="img-card"><img src="test_BoxF1_curve.png" alt="Test BoxF1" /><figcaption>Test Split — Ultralytics F1-Confidence Curve (all classes 0.49 @ 0.325)</figcaption></figure>
        <figure class="img-card"><img src="test_BoxPR_curve.png" alt="Test BoxPR" /><figcaption>Test Split — Ultralytics Precision-Recall Curve (mAP50 = 0.427)</figcaption></figure>
      </div>
    </section>

    <!-- TAB 4: HPARAMS & EPOCH F1 TABLE -->
    <section id="tab-hparams" class="tab-panel">
      <div class="chart-card" style="margin-bottom:18px;">
        <div class="chart-head"><span class="chart-title">TensorBoard HPARAMS — v1 Baseline vs v2 Improved</span></div>
        <div id="hparams-table"></div>
      </div>
      <div class="chart-card">
        <div class="chart-head"><span class="chart-title">v2 YOLO11m @ 960 Epoch-by-Epoch Losses, P/R, F1 Score &amp; mAP (results.csv)</span></div>
        <div id="epoch-table"></div>
      </div>
    </section>
  </main>
</div>

<script>
const DATA = __PAYLOAD_JSON__;

let currentSplit = 'val';
let cmNormalized = false;

const COLORS = {
  Car: '#1f77b4',
  Pedestrian: '#ff7f0e',
  Cyclist: '#2ca02c',
  MicroAll: '#3949ab',
  All: '#3949ab',
  v2: '#ff6f00',
  v1: '#1e88e5'
};

function emaSmooth(vals, w) {
  if (w <= 0) return vals.slice();
  let last = vals[0];
  return vals.map((v, i) => {
    last = last * w + (1 - w) * v;
    const debias = 1 - Math.pow(w, i + 1);
    return last / debias;
  });
}

function drawAxesSVG(svg, xLabel, yLabel, xMin, xMax, yMin, yMax, xTicks, yTicks) {
  const W = 540, H = 320, pad = {l: 52, r: 18, t: 18, b: 42};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const sx = x => pad.l + ((x - xMin) / (xMax - xMin || 1)) * pw;
  const sy = y => pad.t + ph - ((y - yMin) / (yMax - yMin || 1)) * ph;
  let g = `<rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" fill="#fff" stroke="#cfd8dc" stroke-width="1"/>`;
  for (const xt of xTicks) {
    const px = sx(xt);
    g += `<line x1="${px}" y1="${pad.t}" x2="${px}" y2="${pad.t+ph}" stroke="#eeeeee" stroke-width="1"/>`;
    g += `<text x="${px}" y="${pad.t+ph+15}" font-size="10" text-anchor="middle" fill="#616161">${xt}</text>`;
  }
  for (const yt of yTicks) {
    const py = sy(yt);
    g += `<line x1="${pad.l}" y1="${py}" x2="${pad.l+pw}" y2="${py}" stroke="#eeeeee" stroke-width="1"/>`;
    g += `<text x="${pad.l-6}" y="${py+3.5}" font-size="10" text-anchor="end" fill="#616161">${yt}</text>`;
  }
  g += `<text x="${pad.l + pw/2}" y="${H-8}" font-size="11" text-anchor="middle" fill="#424242">${xLabel}</text>`;
  g += `<text x="14" y="${pad.t + ph/2}" font-size="11" text-anchor="middle" fill="#424242" transform="rotate(-90 14 ${pad.t + ph/2})">${yLabel}</text>`;
  return { g, sx, sy, pad, pw, ph };
}

function pathFromPts(pts, sx, sy) {
  if (!pts.length) return '';
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
}

function renderRocTab() {
  const s = DATA.splits[currentSplit];
  const evalM = DATA.eval_v2[currentSplit];
  const evalV1 = DATA.eval_v1[currentSplit];
  const conf = parseFloat(document.getElementById('conf-slider').value);
  const f1Eval = 2 * evalM.precision * evalM.recall / (evalM.precision + evalM.recall);
  const f1V1 = 2 * evalV1.precision * evalV1.recall / (evalV1.precision + evalV1.recall);

  // Scrubber readout
  let rHtml = '';
  for (const cls of ['Car', 'Pedestrian', 'Cyclist', 'MicroAll']) {
    const arr = s.froc_series[cls];
    const nearest = arr.reduce((a, b) => Math.abs(b.conf - conf) < Math.abs(a.conf - conf) ? b : a, arr[0]);
    if (nearest) {
      rHtml += `<div><b style="color:${COLORS[cls]}">${cls.padEnd(10,' ')}</b> F1=${nearest.f1.toFixed(3)} TPR=${nearest.recall.toFixed(3)} P=${nearest.precision.toFixed(3)} FPPI=${nearest.fppi.toFixed(2)}</div>`;
    }
  }
  document.getElementById('conf-readout').innerHTML = rHtml;

  // KPIs
  document.getElementById('roc-kpis').innerHTML = `
    <div class="kpi"><div class="k-lbl">Eval F1 Score (${currentSplit.toUpperCase()})</div><div class="k-val">${f1Eval.toFixed(4)}</div><div class="k-sub">+${(f1Eval-f1V1).toFixed(4)} vs v1 (${f1V1.toFixed(4)})</div></div>
    <div class="kpi"><div class="k-lbl">Peak Macro F1 (BoxF1_curve)</div><div class="k-val">${s.macro_peak_f1.toFixed(3)}</div><div class="k-sub">@ conf = ${s.macro_peak_f1_conf.toFixed(3)}</div></div>
    <div class="kpi"><div class="k-lbl">FROC AUC (0–5 FPPI, Micro)</div><div class="k-val">${s.froc_summary.MicroAll.froc_auc_at_5fppi.toFixed(4)}</div><div class="k-sub">TPR@1 FPPI = ${s.froc_summary.MicroAll.tpr_at_1_fppi.toFixed(3)}</div></div>
    <div class="kpi"><div class="k-lbl">mAP@0.5 / mAP@0.5:0.95</div><div class="k-val">${evalM.map50.toFixed(4)} / ${evalM.map50_95.toFixed(4)}</div><div class="k-sub">P=${evalM.precision.toFixed(3)} · R=${evalM.recall.toFixed(3)}</div></div>
    <div class="kpi"><div class="k-lbl">Matched One-vs-Rest F1 (Car)</div><div class="k-val">${DATA.cm_roc[currentSplit].per_class.Car.ovr_f1.toFixed(4)}</div><div class="k-sub">FPR = ${(DATA.cm_roc[currentSplit].per_class.Car.ovr_fpr*100).toFixed(2)}%</div></div>
  `;

  // 1) FROC Chart (x: FPPI 0..6, y: Recall/TPR 0..1)
  {
    const svg = document.getElementById('svg-froc');
    const ax = drawAxesSVG(svg, 'False Positives Per Image (FPPI)', 'True Positive Rate (Recall)', 0, 6.0, 0, 1.0, [0,1,2,3,4,5,6], [0,0.2,0.4,0.6,0.8,1.0]);
    let inner = ax.g;
    for (const cls of ['Car', 'Pedestrian', 'Cyclist', 'MicroAll']) {
      const pts = s.froc_series[cls].filter(d => d.fppi <= 6.2).sort((a,b) => a.fppi - b.fppi).map(d => [Math.min(6.0, d.fppi), d.recall]);
      inner += `<path d="${pathFromPts(pts, ax.sx, ax.sy)}" fill="none" stroke="${COLORS[cls]}" stroke-width="${cls==='MicroAll'?2.6:1.8}"/>`;
      const near = s.froc_series[cls].reduce((a,b)=>Math.abs(b.conf-conf)<Math.abs(a.conf-conf)?b:a, s.froc_series[cls][0]);
      if (near && near.fppi <= 6.0) {
        inner += `<circle cx="${ax.sx(near.fppi)}" cy="${ax.sy(near.recall)}" r="4" fill="${COLORS[cls]}" stroke="#fff" stroke-width="1.2"/>`;
      }
    }
    svg.innerHTML = inner;
    document.getElementById('froc-stat').textContent = `AUC@5FPPI: ${s.froc_summary.MicroAll.froc_auc_at_5fppi.toFixed(3)} · Marker c=${conf.toFixed(2)}`;
  }

  // 2) F1-Confidence Chart
  {
    const svg = document.getElementById('svg-f1c');
    const ax = drawAxesSVG(svg, 'Confidence Threshold', 'F1 Score', 0, 1.0, 0, 1.0, [0,0.2,0.4,0.6,0.8,1.0], [0,0.2,0.4,0.6,0.8,1.0]);
    let inner = ax.g;
    inner += `<line x1="${ax.sx(conf)}" y1="${ax.pad.t}" x2="${ax.sx(conf)}" y2="${ax.pad.t+ax.ph}" stroke="#9e9e9e" stroke-dasharray="4,3" stroke-width="1.2"/>`;
    for (const cls of ['Car', 'Pedestrian', 'Cyclist']) {
      const pts = s.froc_series[cls].map(d => [d.conf, d.f1]);
      inner += `<path d="${pathFromPts(pts, ax.sx, ax.sy)}" fill="none" stroke="${COLORS[cls]}" stroke-width="1.8"/>`;
    }
    const macroPts = s.macro_f1_curve.map(d => [d.conf, d.f1]);
    inner += `<path d="${pathFromPts(macroPts, ax.sx, ax.sy)}" fill="none" stroke="${COLORS.All}" stroke-width="2.6"/>`;
    svg.innerHTML = inner;
    document.getElementById('f1c-stat').textContent = `Peak Macro F1 = ${s.macro_peak_f1.toFixed(3)} @ c=${s.macro_peak_f1_conf.toFixed(3)}`;
  }

  // 3) PR Chart
  {
    const svg = document.getElementById('svg-pr');
    const ax = drawAxesSVG(svg, 'Recall', 'Precision', 0, 1.0, 0, 1.0, [0,0.2,0.4,0.6,0.8,1.0], [0,0.2,0.4,0.6,0.8,1.0]);
    let inner = ax.g;
    for (const cls of ['Car', 'Pedestrian', 'Cyclist', 'All']) {
      const pts = (s.pr_curves[cls] || []).map(d => [d.recall, d.precision]);
      inner += `<path d="${pathFromPts(pts, ax.sx, ax.sy)}" fill="none" stroke="${COLORS[cls]}" stroke-width="${cls==='All'?2.6:1.8}"/>`;
    }
    svg.innerHTML = inner;
    document.getElementById('pr-stat').textContent = `mAP@0.5 = ${evalM.map50.toFixed(4)} (Car ${evalM.per_class_ap50.Car.toFixed(3)})`;
  }

  // 4) Matched One-vs-Rest ROC Points
  {
    const svg = document.getElementById('svg-ovr');
    const ax = drawAxesSVG(svg, 'False Positive Rate (1 - Specificity)', 'True Positive Rate (One-vs-Rest Recall)', 0, 1.0, 0, 1.0, [0,0.2,0.4,0.6,0.8,1.0], [0,0.2,0.4,0.6,0.8,1.0]);
    let inner = ax.g;
    inner += `<line x1="${ax.sx(0)}" y1="${ax.sy(0)}" x2="${ax.sx(1)}" y2="${ax.sy(1)}" stroke="#bdbdbd" stroke-dasharray="5,4" stroke-width="1.3"/>`;
    const pc = DATA.cm_roc[currentSplit].per_class;
    for (const cls of ['Car', 'Pedestrian', 'Cyclist']) {
      const m = pc[cls];
      // Connect (0,0) -> (FPR, TPR) -> (1,1) operating-point ROC triangle
      const pts = [[0,0], [m.ovr_fpr, m.ovr_tpr], [1,1]];
      inner += `<path d="${pathFromPts(pts, ax.sx, ax.sy)}" fill="none" stroke="${COLORS[cls]}" stroke-width="1.8" stroke-opacity="0.8"/>`;
      inner += `<circle cx="${ax.sx(m.ovr_fpr)}" cy="${ax.sy(m.ovr_tpr)}" r="5.5" fill="${COLORS[cls]}" stroke="#fff" stroke-width="1.5"/>`;
      const lx = Math.min(0.82, m.ovr_fpr + 0.03);
      const ly = Math.max(0.08, m.ovr_tpr - 0.04);
      inner += `<text x="${ax.sx(lx)}" y="${ax.sy(ly)}" font-size="10.5" font-weight="600" fill="${COLORS[cls]}">${cls} (${m.ovr_fpr.toFixed(3)}, ${m.ovr_tpr.toFixed(3)})</text>`;
    }
    svg.innerHTML = inner;
    document.getElementById('ovr-stat').textContent = `Closed-set 3-Class Matched Matrix`;
  }

  // Summary Table
  const fs = s.froc_summary;
  const cm = DATA.cm_roc[currentSplit].per_class;
  let tbl = `<table><thead><tr>
    <th>Class</th><th>GT Count</th><th>AP@0.5</th><th>Peak F1 (@Conf)</th><th>FROC AUC (0–5 FPPI)</th>
    <th>TPR @ 1.0 / 2.0 FPPI</th><th>Matched OvR TPR</th><th>Matched OvR FPR</th><th>Matched OvR F1</th>
  </tr></thead><tbody>`;
  for (const cls of ['Car', 'Pedestrian', 'Cyclist']) {
    tbl += `<tr>
      <td><b style="color:${COLORS[cls]}">${cls}</b></td>
      <td>${s.gt_counts[cls].toLocaleString()}</td>
      <td><b>${evalM.per_class_ap50[cls].toFixed(4)}</b></td>
      <td>${fs[cls].peak_f1.toFixed(3)} (@${fs[cls].peak_f1_conf.toFixed(3)})</td>
      <td>${fs[cls].froc_auc_at_5fppi.toFixed(4)}</td>
      <td>${fs[cls].tpr_at_1_fppi.toFixed(3)} / ${fs[cls].tpr_at_2_fppi.toFixed(3)}</td>
      <td>${cm[cls].ovr_tpr.toFixed(4)}</td>
      <td>${cm[cls].ovr_fpr.toFixed(4)}</td>
      <td><b>${cm[cls].ovr_f1.toFixed(4)}</b></td>
    </tr>`;
  }
  tbl += `<tr class="best-row">
    <td>All / Micro Summary</td>
    <td>${s.gt_counts.All.toLocaleString()}</td>
    <td>${evalM.map50.toFixed(4)}</td>
    <td>Macro ${s.macro_peak_f1.toFixed(3)} (@${s.macro_peak_f1_conf.toFixed(3)})</td>
    <td>${fs.MicroAll.froc_auc_at_5fppi.toFixed(4)}</td>
    <td>${fs.MicroAll.tpr_at_1_fppi.toFixed(3)} / ${fs.MicroAll.tpr_at_2_fppi.toFixed(3)}</td>
    <td colspan="3">Eval Point P=${evalM.precision.toFixed(4)}, R=${evalM.recall.toFixed(4)}, F1=<b>${f1Eval.toFixed(4)}</b></td>
  </tr></tbody></table>`;
  document.getElementById('roc-summary-table').innerHTML = tbl;
  document.getElementById('split-badge').textContent = `${currentSplit.toUpperCase()} · ${s.n_images} frames`;
}

function renderScalarsTab() {
  const showV2 = document.getElementById('chk-v2').checked;
  const showV1 = document.getElementById('chk-v1').checked;
  const smooth = parseFloat(document.getElementById('smooth-slider').value);
  const xMode = document.getElementById('xaxis-select').value;

  const bestV2 = DATA.v2_epochs[11]; // epoch 12
  const bestV1 = DATA.v1_epochs[17]; // epoch 18
  document.getElementById('scalar-kpis').innerHTML = `
    <div class="kpi"><div class="k-lbl">v2 Best Val F1 (Ep 12)</div><div class="k-val">${bestV2['metrics/f1(B)'].toFixed(4)}</div><div class="k-sub">Peak Ep 19: ${Math.max(...DATA.v2_epochs.map(r=>r['metrics/f1(B)'])).toFixed(4)}</div></div>
    <div class="kpi"><div class="k-lbl">v2 Best mAP@0.5 (Ep 12)</div><div class="k-val">${bestV2['metrics/mAP50(B)'].toFixed(4)}</div><div class="k-sub">vs v1 Best ${bestV1['metrics/mAP50(B)'].toFixed(4)} (+15.7%)</div></div>
    <div class="kpi"><div class="k-lbl">v2 Best mAP@0.5:0.95</div><div class="k-val">${bestV2['metrics/mAP50-95(B)'].toFixed(4)}</div><div class="k-sub">vs v1 Best ${bestV1['metrics/mAP50-95(B)'].toFixed(4)} (+23.8%)</div></div>
    <div class="kpi"><div class="k-lbl">v2 Precision / Recall (Ep 12)</div><div class="k-val">${bestV2['metrics/precision(B)'].toFixed(3)} / ${bestV2['metrics/recall(B)'].toFixed(3)}</div><div class="k-sub">24 ep early stop (~88 min)</div></div>
  `;

  const groups = [
    ['grp-metrics', ['metrics/f1(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']],
    ['grp-train', ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']],
    ['grp-val', ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']],
    ['grp-lr', ['lr/pg0', 'lr/pg1', 'lr/pg2']]
  ];

  for (const [elId, tags] of groups) {
    const container = document.getElementById(elId);
    container.innerHTML = '';
    for (const tag of tags) {
      const card = document.createElement('div');
      card.className = 'chart-card';
      const v2Last = DATA.v2_epochs[DATA.v2_epochs.length - 1][tag];
      const v2Best = tag.includes('loss') ? Math.min(...DATA.v2_epochs.map(r=>r[tag])) : Math.max(...DATA.v2_epochs.map(r=>r[tag]));
      card.innerHTML = `<div class="chart-head"><span class="chart-title">${tag}</span><span class="chart-stat">v2 best=${v2Best.toFixed(4)} · last=${v2Last.toFixed(4)}</span></div><svg class="plot" viewBox="0 0 540 260"></svg>`;
      container.appendChild(card);
      const svg = card.querySelector('svg');

      const allVals = [];
      const xVals = [];
      if (showV2) DATA.v2_epochs.forEach(r => { allVals.push(r[tag]); xVals.push(xMode==='epoch'?r.epoch:r.time/60); });
      if (showV1) DATA.v1_epochs.forEach(r => { allVals.push(r[tag]); xVals.push(xMode==='epoch'?r.epoch:r.time/60); });
      if (!allVals.length) continue;

      let yMin = Math.min(...allVals), yMax = Math.max(...allVals);
      const yPad = (yMax - yMin) * 0.1 || 0.05;
      yMin = Math.max(0, yMin - yPad); yMax = yMax + yPad;
      const xMin = 0, xMax = Math.ceil(Math.max(...xVals));
      const xTicks = Array.from({length:6}, (_,i)=> +(xMin + (xMax-xMin)*i/5).toFixed(1));
      const yTicks = Array.from({length:5}, (_,i)=> +(yMin + (yMax-yMin)*i/4).toFixed(tag.startsWith('lr/')?5:3));

      const W = 540, H = 260, pad = {l: 56, r: 16, t: 14, b: 36};
      const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
      const sx = x => pad.l + ((x - xMin)/(xMax - xMin || 1))*pw;
      const sy = y => pad.t + ph - ((y - yMin)/(yMax - yMin || 1))*ph;
      let g = `<rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" fill="#fff" stroke="#cfd8dc"/>`;
      xTicks.forEach(t => { g += `<line x1="${sx(t)}" y1="${pad.t}" x2="${sx(t)}" y2="${pad.t+ph}" stroke="#f0f0f0"/><text x="${sx(t)}" y="${pad.t+ph+14}" font-size="10" text-anchor="middle" fill="#616161">${t}</text>`; });
      yTicks.forEach(t => { g += `<line x1="${pad.l}" y1="${sy(t)}" x2="${pad.l+pw}" y2="${sy(t)}" stroke="#f0f0f0"/><text x="${pad.l-5}" y="${sy(t)+3}" font-size="10" text-anchor="end" fill="#616161">${t}</text>`; });
      g += `<text x="${pad.l+pw/2}" y="${H-6}" font-size="10" text-anchor="middle" fill="#616161">${xMode==='epoch'?'Epoch':'Elapsed Time (min)'}</text>`;

      const plotRun = (rows, col) => {
        const xs = rows.map(r => xMode==='epoch'?r.epoch:r.time/60);
        const raw = rows.map(r => r[tag]);
        const sm = emaSmooth(raw, smooth);
        const rawPts = xs.map((x,i)=>[x, raw[i]]);
        const smPts = xs.map((x,i)=>[x, sm[i]]);
        let out = `<path d="${pathFromPts(rawPts, sx, sy)}" fill="none" stroke="${col}" stroke-opacity="0.28" stroke-width="1.3"/>`;
        out += `<path d="${pathFromPts(smPts, sx, sy)}" fill="none" stroke="${col}" stroke-width="2.2"/>`;
        return out;
      };
      if (showV1) g += plotRun(DATA.v1_epochs, COLORS.v1);
      if (showV2) g += plotRun(DATA.v2_epochs, COLORS.v2);
      svg.innerHTML = g;
    }
  }
}

function renderConfusionMatrices() {
  const labels = ['Car', 'Pedestrian', 'Cyclist', 'bg'];
  for (const [elId, split] of [['cm-val-box', 'val'], ['cm-test-box', 'test']]) {
    const mat = DATA.cm_roc[split].matrix;
    const colSums = [0,1,2,3].map(c => mat.reduce((s,row)=>s+row[c], 0) || 1);
    const maxVal = Math.max(...mat.flat());
    let html = `<div class="cm-grid">`;
    html += `<div class="cm-cell cm-hdr">Pred \\ True</div>`;
    labels.forEach(l => html += `<div class="cm-cell cm-hdr">${l}</div>`);
    mat.forEach((row, r) => {
      html += `<div class="cm-cell cm-hdr">${labels[r]}</div>`;
      row.forEach((v, c) => {
        const ratio = cmNormalized ? (v / colSums[c]) : (v / maxVal);
        const bg = v === 0 ? '#fafafa' : `rgba(25, 118, 210, ${(0.08 + 0.78 * Math.min(1, ratio)).toFixed(2)})`;
        const fg = ratio > 0.45 ? '#fff' : '#212121';
        const txt = cmNormalized ? `${(100*v/colSums[c]).toFixed(1)}%` : v.toLocaleString();
        html += `<div class="cm-cell" style="background:${bg}; color:${fg}; font-weight:${r===c?'700':'400'}">${txt}</div>`;
      });
    });
    html += `</div>`;
    document.getElementById(elId).innerHTML = html;
  }
}

function toggleCmNorm() {
  cmNormalized = !cmNormalized;
  renderConfusionMatrices();
}

function renderHparamsTab() {
  const keys = ['model', 'imgsz', 'epochs', 'patience', 'batch', 'lr0', 'lrf', 'cos_lr', 'mixup', 'close_mosaic', 'box', 'cls', 'dfl', 'oversample_rare_classes'];
  let hp = `<table><thead><tr><th>Hyperparameter / Metric</th><th>v1 Baseline</th><th>v2 Improved</th></tr></thead><tbody>`;
  keys.forEach(k => {
    const v1v = k === 'oversample_rare_classes' ? 'false' : (DATA.v1_args[k] ?? '—');
    const v2v = k === 'oversample_rare_classes' ? 'true (Cyc x4, Ped x2)' : (DATA.v2_args[k] ?? '—');
    hp += `<tr><td><code>${k}</code></td><td>${v1v}</td><td><b>${v2v}</b></td></tr>`;
  });
  const f1v1 = 2*DATA.eval_v1.val.precision*DATA.eval_v1.val.recall/(DATA.eval_v1.val.precision+DATA.eval_v1.val.recall);
  const f1v2 = 2*DATA.eval_v2.val.precision*DATA.eval_v2.val.recall/(DATA.eval_v2.val.precision+DATA.eval_v2.val.recall);
  hp += `<tr class="best-row"><td>Val F1 Score / mAP@0.5</td><td>${f1v1.toFixed(4)} / ${DATA.eval_v1.val.map50.toFixed(4)}</td><td>${f1v2.toFixed(4)} / ${DATA.eval_v2.val.map50.toFixed(4)}</td></tr>`;
  hp += `</tbody></table>`;
  document.getElementById('hparams-table').innerHTML = hp;

  let ep = `<table><thead><tr>
    <th>Epoch</th><th>Time (min)</th><th>train/box</th><th>train/cls</th><th>train/dfl</th>
    <th>Precision</th><th>Recall</th><th><b>F1 Score</b></th><th>mAP@0.5</th><th>mAP@0.5:0.95</th>
  </tr></thead><tbody>`;
  DATA.v2_epochs.forEach(r => {
    const isBest = r.epoch === 12;
    ep += `<tr class="${isBest?'best-row':''}">
      <td>${r.epoch}${isBest?' (Best)':''}</td>
      <td>${(r.time/60).toFixed(1)}</td>
      <td>${r['train/box_loss'].toFixed(4)}</td>
      <td>${r['train/cls_loss'].toFixed(4)}</td>
      <td>${r['train/dfl_loss'].toFixed(4)}</td>
      <td>${r['metrics/precision(B)'].toFixed(4)}</td>
      <td>${r['metrics/recall(B)'].toFixed(4)}</td>
      <td><b>${r['metrics/f1(B)'].toFixed(4)}</b></td>
      <td>${r['metrics/mAP50(B)'].toFixed(4)}</td>
      <td>${r['metrics/mAP50-95(B)'].toFixed(4)}</td>
    </tr>`;
  });
  ep += `</tbody></table>`;
  document.getElementById('epoch-table').innerHTML = ep;
}

// Wire controls
document.querySelectorAll('nav.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

document.querySelectorAll('[data-split]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-split]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSplit = btn.dataset.split;
    renderRocTab();
  });
});

document.getElementById('conf-slider').addEventListener('input', e => {
  document.getElementById('conf-val').textContent = parseFloat(e.target.value).toFixed(3);
  renderRocTab();
});

document.getElementById('smooth-slider').addEventListener('input', e => {
  document.getElementById('smooth-val').textContent = parseFloat(e.target.value).toFixed(2);
  renderScalarsTab();
});

['chk-v2', 'chk-v1', 'xaxis-select'].forEach(id => {
  document.getElementById(id).addEventListener('change', renderScalarsTab);
});

renderRocTab();
renderScalarsTab();
renderConfusionMatrices();
renderHparamsTab();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Vector LaTeX + PGFPlots Summary Builder
# ---------------------------------------------------------------------------

def build_tex_and_pdf(
    v2_rows: List[Dict[str, float]],
    eval_v2: Dict[str, Any],
    eval_v1: Dict[str, Any],
    val_curves: Dict[str, Any],
    test_curves: Dict[str, Any],
    val_cm: Dict[str, Any],
    test_cm: Dict[str, Any],
) -> None:
    def _coords_xy(items: List[Tuple[float, float]]) -> str:
        return " ".join(f"({x:.4f},{y:.4f})" for x, y in items)

    # Downsample ~55 points cleanly for PGFPlots
    def _subsample(seq: List[Dict[str, float]], xk: str, yk: str, max_x: Optional[float] = None) -> List[Tuple[float, float]]:
        pts = [(min(max_x, d[xk]) if max_x is not None else d[xk], d[yk]) for d in seq]
        if len(pts) <= 65:
            return pts
        step = max(1, len(pts) // 60)
        return pts[::step] + [pts[-1]]

    f1_val_macro = _coords_xy(_subsample(val_curves["macro_f1_curve"], "conf", "f1"))
    f1_val_car = _coords_xy(_subsample(val_curves["froc_series"]["Car"], "conf", "f1"))
    f1_val_ped = _coords_xy(_subsample(val_curves["froc_series"]["Pedestrian"], "conf", "f1"))
    f1_val_cyc = _coords_xy(_subsample(val_curves["froc_series"]["Cyclist"], "conf", "f1"))

    froc_val_car = _coords_xy(_subsample(sorted(val_curves["froc_series"]["Car"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_val_ped = _coords_xy(_subsample(sorted(val_curves["froc_series"]["Pedestrian"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_val_cyc = _coords_xy(_subsample(sorted(val_curves["froc_series"]["Cyclist"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_val_all = _coords_xy(_subsample(sorted(val_curves["froc_series"]["MicroAll"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))

    froc_test_car = _coords_xy(_subsample(sorted(test_curves["froc_series"]["Car"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_test_ped = _coords_xy(_subsample(sorted(test_curves["froc_series"]["Pedestrian"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_test_cyc = _coords_xy(_subsample(sorted(test_curves["froc_series"]["Cyclist"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))
    froc_test_all = _coords_xy(_subsample(sorted(test_curves["froc_series"]["MicroAll"], key=lambda d: d["fppi"]), "fppi", "recall", 6.0))

    ep_f1 = _coords_xy([(r["epoch"], r["metrics/f1(B)"]) for r in v2_rows])
    ep_map50 = _coords_xy([(r["epoch"], r["metrics/mAP50(B)"]) for r in v2_rows])
    ep_p = _coords_xy([(r["epoch"], r["metrics/precision(B)"]) for r in v2_rows])
    ep_r = _coords_xy([(r["epoch"], r["metrics/recall(B)"]) for r in v2_rows])

    val_f1_eval = 2 * eval_v2["val"]["precision"] * eval_v2["val"]["recall"] / (eval_v2["val"]["precision"] + eval_v2["val"]["recall"])
    test_f1_eval = 2 * eval_v2["test"]["precision"] * eval_v2["test"]["recall"] / (eval_v2["test"]["precision"] + eval_v2["test"]["recall"])

    tex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=15mm]{{geometry}}
\usepackage{{kotex}}
\usepackage{{fontspec}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{pgfplots}}
\usetikzlibrary{{calc}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\pgfplotsset{{compat=1.18}}
\setmainfont{{Apple SD Gothic Neo}}[BoldFont={{* Bold}}]
\setmonofont{{Menlo}}[Scale=0.84]
\definecolor{{tborange}}{{HTML}}{{FF6F00}}
\definecolor{{cblue}}{{HTML}}{{1F77B4}}
\definecolor{{corange}}{{HTML}}{{FF7F0E}}
\definecolor{{cgreen}}{{HTML}}{{2CA02C}}
\definecolor{{call}}{{HTML}}{{3949AB}}

\title{{\textbf{{TensorBoard 기준 학습·평가 지표 보고서 (ROC/FROC · F1 Score · Scalars)}}}}
\author{{Totalcare KITTI YOLO11m @ 960 (v2\_improved)}}
\date{{2026-11-14}}

\begin{{document}}
\maketitle
\vspace{{-8mm}}

\noindent\textbf{{대상 디렉터리:}} \texttt{{results/v2\_improved/outputs/train\_results/}} \\
\textbf{{원본 실행:}} Kaggle 2$\times$ Tesla T4 DDP (\texttt{{epochs=40}}, Early Stop \texttt{{ep=24}}, Best \texttt{{ep=12}}) · Val \texttt{{832}} imgs / Test \texttt{{1,608}} imgs

\section*{{1. TensorBoard SCALARS / TIME SERIES 요약 및 F1 Score 추이}}
\begin{{itemize}}
  \item \textbf{{Val Eval Operating Point (\texttt{{best.pt}}):}} Precision \textbf{{{eval_v2['val']['precision']:.4f}}}, Recall \textbf{{{eval_v2['val']['recall']:.4f}}}, \textbf{{F1 = {val_f1_eval:.4f}}} (v1 Baseline \texttt{{0.4971}} 대비 \textbf{{+{val_f1_eval - 2*eval_v1['val']['precision']*eval_v1['val']['recall']/(eval_v1['val']['precision']+eval_v1['val']['recall']):.4f}}}), mAP@0.5 = \textbf{{{eval_v2['val']['map50']:.4f}}}
  \item \textbf{{Test Eval Operating Point (Held-out 0013/0016/0019):}} Precision \textbf{{{eval_v2['test']['precision']:.4f}}}, Recall \textbf{{{eval_v2['test']['recall']:.4f}}}, \textbf{{F1 = {test_f1_eval:.4f}}}, mAP@0.5 = \textbf{{{eval_v2['test']['map50']:.4f}}}
  \item \textbf{{Peak F1 on Confidence Sweep:}} Val \textbf{{{val_curves['macro_peak_f1']:.3f}}} @ conf \texttt{{{val_curves['macro_peak_f1_conf']:.3f}}}, Test \textbf{{{test_curves['macro_peak_f1']:.3f}}} @ conf \texttt{{{test_curves['macro_peak_f1_conf']:.3f}}}
\end{{itemize}}

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.92\textwidth, height=5.4cm,
  title={{Epoch별 F1 Score · Precision · Recall · mAP@0.5 (v2 YOLO11m @ 960)}},
  xlabel={{Epoch}}, ylabel={{Score}},
  xmin=1, xmax=24, ymin=0.25, ymax=0.85,
  grid=both, legend pos=south east, legend columns=4,
  tick label style={{font=\footnotesize}}, label style={{font=\small}}
]
\addplot[color=tborange, thick, mark=*, mark size=1.3pt] coordinates {{{ep_f1}}}; \addlegendentry{{F1(B)}}
\addplot[color=cblue, thick, dashed] coordinates {{{ep_p}}}; \addlegendentry{{Precision(B)}}
\addplot[color=cgreen, thick, dotted] coordinates {{{ep_r}}}; \addlegendentry{{Recall(B)}}
\addplot[color=call, thick] coordinates {{{ep_map50}}}; \addlegendentry{{mAP@0.5(B)}}
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}

\section*{{2. Detection FROC (ROC 대응 곡선) · F1-Confidence · 1-vs-Rest ROC 분석}}
객체 검출기에서는 배경(True Negative) 영역이 무한대이므로 순수 box-level \texttt{{TN}}이 정의되지 않습니다. 따라서 TensorBoard 및 자율주행 표준에 따라
\textbf{{(A) Detection FROC Curve (Sensitivity/TPR vs False Positives Per Image [FPPI])}},
\textbf{{(B) F1-Confidence Curve}},
\textbf{{(C) Precision-Recall (AP50)}},
\textbf{{(D) Confusion-Matrix One-vs-Rest ROC (\texttt{{TPR vs FPR}})}}로 정량 분석을 제공합니다.

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
  name=ax1, width=0.47\textwidth, height=5.5cm,
  title={{Val Detection FROC (TPR vs FPPI)}},
  xlabel={{False Positives Per Image (FPPI)}}, ylabel={{True Positive Rate (Recall)}},
  xmin=0, xmax=6, ymin=0, ymax=1.0, grid=both,
  legend pos=north east, tick label style={{font=\scriptsize}}, label style={{font=\footnotesize}}
]
\addplot[color=cblue, thick] coordinates {{{froc_val_car}}}; \addlegendentry{{Car}}
\addplot[color=corange, thick] coordinates {{{froc_val_ped}}}; \addlegendentry{{Pedestrian}}
\addplot[color=cgreen, thick] coordinates {{{froc_val_cyc}}}; \addlegendentry{{Cyclist}}
\addplot[color=call, very thick] coordinates {{{froc_val_all}}}; \addlegendentry{{Micro-All}}
\end{{axis}}

\begin{{axis}}[
  at={{($(ax1.outer south east)+(0.5cm,0)$)}}, anchor=outer south west,
  width=0.47\textwidth, height=5.5cm,
  title={{Test Detection FROC (TPR vs FPPI)}},
  xlabel={{False Positives Per Image (FPPI)}}, ylabel={{True Positive Rate (Recall)}},
  xmin=0, xmax=6, ymin=0, ymax=1.0, grid=both,
  legend pos=north east, tick label style={{font=\scriptsize}}, label style={{font=\footnotesize}}
]
\addplot[color=cblue, thick] coordinates {{{froc_test_car}}}; \addlegendentry{{Car}}
\addplot[color=corange, thick] coordinates {{{froc_test_ped}}}; \addlegendentry{{Pedestrian}}
\addplot[color=cgreen, thick] coordinates {{{froc_test_cyc}}}; \addlegendentry{{Cyclist}}
\addplot[color=call, very thick] coordinates {{{froc_test_all}}}; \addlegendentry{{Micro-All}}
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
  name=ax3, width=0.47\textwidth, height=5.2cm,
  title={{Val F1-Confidence Curve}},
  xlabel={{Confidence Threshold}}, ylabel={{F1 Score}},
  xmin=0, xmax=1, ymin=0, ymax=1.0, grid=both,
  legend pos=north east, tick label style={{font=\scriptsize}}, label style={{font=\footnotesize}}
]
\addplot[color=cblue, thick] coordinates {{{f1_val_car}}}; \addlegendentry{{Car}}
\addplot[color=corange, thick] coordinates {{{f1_val_ped}}}; \addlegendentry{{Pedestrian}}
\addplot[color=cgreen, thick] coordinates {{{f1_val_cyc}}}; \addlegendentry{{Cyclist}}
\addplot[color=call, very thick] coordinates {{{f1_val_macro}}}; \addlegendentry{{All ({val_curves['macro_peak_f1']:.2f}@{val_curves['macro_peak_f1_conf']:.2f})}}
\end{{axis}}

\begin{{axis}}[
  at={{($(ax3.outer south east)+(0.5cm,0)$)}}, anchor=outer south west,
  width=0.47\textwidth, height=5.2cm,
  title={{Matched 1-vs-Rest ROC Points (Val \& Test)}},
  xlabel={{False Positive Rate (FPR)}}, ylabel={{True Positive Rate (TPR)}},
  xmin=0, xmax=1, ymin=0, ymax=1.0, grid=both,
  legend pos=south east, tick label style={{font=\scriptsize}}, label style={{font=\footnotesize}}
]
\addplot[color=gray!60, dashed] coordinates {{(0,0) (1,1)}}; \addlegendentry{{Chance}}
\addplot[color=cblue, only marks, mark=*, mark size=2.3pt] coordinates {{({val_cm['per_class']['Car']['ovr_fpr']},{val_cm['per_class']['Car']['ovr_tpr']})}}; \addlegendentry{{Val Car}}
\addplot[color=corange, only marks, mark=*, mark size=2.3pt] coordinates {{({val_cm['per_class']['Pedestrian']['ovr_fpr']},{val_cm['per_class']['Pedestrian']['ovr_tpr']})}}; \addlegendentry{{Val Ped}}
\addplot[color=cgreen, only marks, mark=*, mark size=2.3pt] coordinates {{({val_cm['per_class']['Cyclist']['ovr_fpr']},{val_cm['per_class']['Cyclist']['ovr_tpr']})}}; \addlegendentry{{Val Cyc}}
\addplot[color=cblue, only marks, mark=square*, mark size=2.3pt] coordinates {{({test_cm['per_class']['Car']['ovr_fpr']},{test_cm['per_class']['Car']['ovr_tpr']})}}; \addlegendentry{{Test Car}}
\addplot[color=corange, only marks, mark=square*, mark size=2.3pt] coordinates {{({test_cm['per_class']['Pedestrian']['ovr_fpr']},{test_cm['per_class']['Pedestrian']['ovr_tpr']})}}; \addlegendentry{{Test Ped}}
\addplot[color=cgreen, only marks, mark=square*, mark size=2.3pt] coordinates {{({test_cm['per_class']['Cyclist']['ovr_fpr']},{test_cm['per_class']['Cyclist']['ovr_tpr']})}}; \addlegendentry{{Test Cyc}}
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}

\section*{{3. 클래스별 F1 · FROC AUC · 1-vs-Rest ROC 정량 지표 표}}
\begin{{center}}
\small
\begin{{tabular}}{{llrrrrrrr}}
\toprule
\textbf{{Split}} & \textbf{{Class}} & \textbf{{GT}} & \textbf{{AP@0.5}} & \textbf{{Peak F1 (@Conf)}} & \textbf{{FROC AUC (5 FPPI)}} & \textbf{{TPR@1/2 FPPI}} & \textbf{{OvR TPR}} & \textbf{{OvR FPR}} \\
\midrule
Val  & Car        & 2,075 & {eval_v2['val']['per_class_ap50']['Car']:.4f} & {val_curves['froc_summary']['Car']['peak_f1']:.3f} (@{val_curves['froc_summary']['Car']['peak_f1_conf']:.2f}) & {val_curves['froc_summary']['Car']['froc_auc_at_5fppi']:.4f} & {val_curves['froc_summary']['Car']['tpr_at_1_fppi']:.3f} / {val_curves['froc_summary']['Car']['tpr_at_2_fppi']:.3f} & {val_cm['per_class']['Car']['ovr_tpr']:.4f} & {val_cm['per_class']['Car']['ovr_fpr']:.4f} \\
Val  & Pedestrian & 1,777 & {eval_v2['val']['per_class_ap50']['Pedestrian']:.4f} & {val_curves['froc_summary']['Pedestrian']['peak_f1']:.3f} (@{val_curves['froc_summary']['Pedestrian']['peak_f1_conf']:.2f}) & {val_curves['froc_summary']['Pedestrian']['froc_auc_at_5fppi']:.4f} & {val_curves['froc_summary']['Pedestrian']['tpr_at_1_fppi']:.3f} / {val_curves['froc_summary']['Pedestrian']['tpr_at_2_fppi']:.3f} & {val_cm['per_class']['Pedestrian']['ovr_tpr']:.4f} & {val_cm['per_class']['Pedestrian']['ovr_fpr']:.4f} \\
Val  & Cyclist    & 754   & {eval_v2['val']['per_class_ap50']['Cyclist']:.4f} & {val_curves['froc_summary']['Cyclist']['peak_f1']:.3f} (@{val_curves['froc_summary']['Cyclist']['peak_f1_conf']:.2f}) & {val_curves['froc_summary']['Cyclist']['froc_auc_at_5fppi']:.4f} & {val_curves['froc_summary']['Cyclist']['tpr_at_1_fppi']:.3f} / {val_curves['froc_summary']['Cyclist']['tpr_at_2_fppi']:.3f} & {val_cm['per_class']['Cyclist']['ovr_tpr']:.4f} & {val_cm['per_class']['Cyclist']['ovr_fpr']:.4f} \\
Val  & \textbf{{All / Micro}} & 4,606 & \textbf{{{eval_v2['val']['map50']:.4f}}} & Macro {val_curves['macro_peak_f1']:.3f} (@{val_curves['macro_peak_f1_conf']:.3f}) & {val_curves['froc_summary']['MicroAll']['froc_auc_at_5fppi']:.4f} & {val_curves['froc_summary']['MicroAll']['tpr_at_1_fppi']:.3f} / {val_curves['froc_summary']['MicroAll']['tpr_at_2_fppi']:.3f} & \multicolumn{{2}}{{r}}{{Eval F1 = \textbf{{{val_f1_eval:.4f}}}}} \\
\midrule
Test & Car        & 1,818 & {eval_v2['test']['per_class_ap50']['Car']:.4f} & {test_curves['froc_summary']['Car']['peak_f1']:.3f} (@{test_curves['froc_summary']['Car']['peak_f1_conf']:.2f}) & {test_curves['froc_summary']['Car']['froc_auc_at_5fppi']:.4f} & {test_curves['froc_summary']['Car']['tpr_at_1_fppi']:.3f} / {test_curves['froc_summary']['Car']['tpr_at_2_fppi']:.3f} & {test_cm['per_class']['Car']['ovr_tpr']:.4f} & {test_cm['per_class']['Car']['ovr_fpr']:.4f} \\
Test & Pedestrian & 9,044 & {eval_v2['test']['per_class_ap50']['Pedestrian']:.4f} & {test_curves['froc_summary']['Pedestrian']['peak_f1']:.3f} (@{test_curves['froc_summary']['Pedestrian']['peak_f1_conf']:.2f}) & {test_curves['froc_summary']['Pedestrian']['froc_auc_at_5fppi']:.4f} & {test_curves['froc_summary']['Pedestrian']['tpr_at_1_fppi']:.3f} / {test_curves['froc_summary']['Pedestrian']['tpr_at_2_fppi']:.3f} & {test_cm['per_class']['Pedestrian']['ovr_tpr']:.4f} & {test_cm['per_class']['Pedestrian']['ovr_fpr']:.4f} \\
Test & Cyclist    & 817   & {eval_v2['test']['per_class_ap50']['Cyclist']:.4f} & {test_curves['froc_summary']['Cyclist']['peak_f1']:.3f} (@{test_curves['froc_summary']['Cyclist']['peak_f1_conf']:.2f}) & {test_curves['froc_summary']['Cyclist']['froc_auc_at_5fppi']:.4f} & {test_curves['froc_summary']['Cyclist']['tpr_at_1_fppi']:.3f} / {test_curves['froc_summary']['Cyclist']['tpr_at_2_fppi']:.3f} & {test_cm['per_class']['Cyclist']['ovr_tpr']:.4f} & {test_cm['per_class']['Cyclist']['ovr_fpr']:.4f} \\
Test & \textbf{{All / Micro}} & 11,679 & \textbf{{{eval_v2['test']['map50']:.4f}}} & Macro {test_curves['macro_peak_f1']:.3f} (@{test_curves['macro_peak_f1_conf']:.3f}) & {test_curves['froc_summary']['MicroAll']['froc_auc_at_5fppi']:.4f} & {test_curves['froc_summary']['MicroAll']['tpr_at_1_fppi']:.3f} / {test_curves['froc_summary']['MicroAll']['tpr_at_2_fppi']:.3f} & \multicolumn{{2}}{{r}}{{Eval F1 = \textbf{{{test_f1_eval:.4f}}}}} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\end{{document}}
"""
    tex_path = V2_TRAIN_DIR / "roc_f1_tensorboard_summary.tex"
    tex_path.write_text(tex, encoding="utf-8")

    tectonic = shutil.which("tectonic") or str(Path.home() / ".local" / "bin" / "tectonic")
    if Path(tectonic).exists():
        subprocess.run([tectonic, str(tex_path), "--outdir", str(V2_TRAIN_DIR)], check=True)
    else:
        print("[WARN] tectonic not found; skipped PDF compile")


def main() -> None:
    V2_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Copy native Ultralytics Val & Test curves directly into train_results/ so they sit next to results.csv
    for fname in ("BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"):
        if (V2_VAL_RUN_DIR / fname).exists():
            shutil.copy2(V2_VAL_RUN_DIR / fname, V2_TRAIN_DIR / fname)
        if (V2_TEST_RUN_DIR / fname).exists():
            shutil.copy2(V2_TEST_RUN_DIR / fname, V2_TRAIN_DIR / f"test_{fname}")

    v2_rows = load_results_csv(V2_TRAIN_DIR / "results.csv")
    v1_rows = load_results_csv(V1_TRAIN_DIR / "results.csv")
    v2_args = load_simple_yaml(V2_TRAIN_DIR / "args.yaml")
    v1_args = load_simple_yaml(V1_TRAIN_DIR / "args.yaml")
    eval_v2 = json.loads((V2_OUTPUT_DIR / "eval_metrics.json").read_text(encoding="utf-8"))
    eval_v1 = json.loads((V1_OUTPUT_DIR / "eval_metrics.json").read_text(encoding="utf-8"))

    val_curves = build_split_curves(V2_VAL_RUN_DIR, "val")
    test_curves = build_split_curves(V2_TEST_RUN_DIR, "test")
    val_cm = compute_confusion_roc_metrics("val")
    test_cm = compute_confusion_roc_metrics("test")

    # Save epoch_f1_summary.csv
    epoch_csv_path = V2_TRAIN_DIR / "epoch_f1_summary.csv"
    with epoch_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch", "time_sec", "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/f1(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)", "lr/pg0",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in v2_rows:
            writer.writerow({
                "epoch": int(r["epoch"]),
                "time_sec": round(r["time"], 2),
                "train/box_loss": r["train/box_loss"],
                "train/cls_loss": r["train/cls_loss"],
                "train/dfl_loss": r["train/dfl_loss"],
                "val/box_loss": r["val/box_loss"],
                "val/cls_loss": r["val/cls_loss"],
                "val/dfl_loss": r["val/dfl_loss"],
                "metrics/precision(B)": r["metrics/precision(B)"],
                "metrics/recall(B)": r["metrics/recall(B)"],
                "metrics/f1(B)": r["metrics/f1(B)"],
                "metrics/mAP50(B)": r["metrics/mAP50(B)"],
                "metrics/mAP50-95(B)": r["metrics/mAP50-95(B)"],
                "lr/pg0": r["lr/pg0"],
            })

    payload = {
        "v2_epochs": v2_rows,
        "v1_epochs": v1_rows,
        "v2_args": v2_args,
        "v1_args": v1_args,
        "eval_v2": eval_v2,
        "eval_v1": eval_v1,
        "splits": {"val": val_curves, "test": test_curves},
        "cm_roc": {"val": val_cm, "test": test_cm},
    }
    (V2_TRAIN_DIR / "roc_f1_curve_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write authentic TFRecord TensorBoard event files
    tb_v2, tb_v1 = write_tfevents(v2_rows, v1_rows, eval_v2, val_curves, test_curves)

    # Write TensorBoard-style interactive HTML dashboard
    html = DASHBOARD_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    (V2_TRAIN_DIR / "tensorboard_dashboard.html").write_text(html, encoding="utf-8")

    # Write Markdown summary
    val_f1 = 2 * eval_v2["val"]["precision"] * eval_v2["val"]["recall"] / (eval_v2["val"]["precision"] + eval_v2["val"]["recall"])
    test_f1 = 2 * eval_v2["test"]["precision"] * eval_v2["test"]["recall"] / (eval_v2["test"]["precision"] + eval_v2["test"]["recall"])
    md_lines = [
        "# TensorBoard 기준 학습·평가 지표 요약 (ROC / FROC · F1 Score · Scalars)",
        "",
        "- **대상 경로**: `results/v2_improved/outputs/train_results/`",
        f"- **실제 TFRecord 이벤트 로그**: `tensorboard --logdir {V2_TRAIN_DIR / 'tensorboard_logs'}`",
        f"- **인터랙티브 대시보드**: `tensorboard_dashboard.html` (SCALARS, ROC/FROC & F1, CONFUSION MATRIX, HPARAMS 탭 구성)",
        "",
        "## 1. F1 Score 요약 (Eval Point + Confidence Sweep Peak)",
        "| Split | Precision | Recall | **Eval F1** | Peak Macro F1 (@Conf) | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **Val (832 imgs)** | {eval_v2['val']['precision']:.4f} | {eval_v2['val']['recall']:.4f} | **{val_f1:.4f}** | **{val_curves['macro_peak_f1']:.3f}** (@{val_curves['macro_peak_f1_conf']:.3f}) | {eval_v2['val']['map50']:.4f} | {eval_v2['val']['map50_95']:.4f} |",
        f"| **Test (1,608 imgs)** | {eval_v2['test']['precision']:.4f} | {eval_v2['test']['recall']:.4f} | **{test_f1:.4f}** | **{test_curves['macro_peak_f1']:.3f}** (@{test_curves['macro_peak_f1_conf']:.3f}) | {eval_v2['test']['map50']:.4f} | {eval_v2['test']['map50_95']:.4f} |",
        "",
        "## 2. Detection FROC (TPR vs FPPI) & One-vs-Rest Confusion-Matrix ROC",
        "> 객체 검출에서는 배경(True Negative) 박스 수가 무한대이므로 검출기 표준 ROC 대응 지표인 **FROC (TPR vs 이미지당 오경보 FPPI)**와 매칭된 3-클래스 분류 행렬 기반 **One-vs-Rest ROC (TPR vs FPR)**를 함께 제공합니다.",
        "",
        "| Split | Class | Peak F1 (@Conf) | FROC AUC (0–5 FPPI) | TPR @ 0.5 / 1.0 / 2.0 FPPI | OvR TPR | OvR FPR | OvR F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sp, curves, cm in [("Val", val_curves, val_cm), ("Test", test_curves, test_cm)]:
        for cls in CLASSES:
            fs = curves["froc_summary"][cls]
            oc = cm["per_class"][cls]
            md_lines.append(
                f"| {sp} | {cls} | {fs['peak_f1']:.3f} (@{fs['peak_f1_conf']:.3f}) | {fs['froc_auc_at_5fppi']:.4f} | "
                f"{fs['tpr_at_0_5_fppi']:.3f} / {fs['tpr_at_1_fppi']:.3f} / {fs['tpr_at_2_fppi']:.3f} | "
                f"{oc['ovr_tpr']:.4f} | {oc['ovr_fpr']:.4f} | {oc['ovr_f1']:.4f} |"
            )
        ma = curves["froc_summary"]["MicroAll"]
        md_lines.append(
            f"| {sp} | **Micro-All** | {ma['peak_f1']:.3f} (@{ma['peak_f1_conf']:.3f}) | {ma['froc_auc_at_5fppi']:.4f} | "
            f"{ma['tpr_at_0_5_fppi']:.3f} / {ma['tpr_at_1_fppi']:.3f} / {ma['tpr_at_2_fppi']:.3f} | — | — | — |"
        )
    (V2_TRAIN_DIR / "roc_f1_tensorboard_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Vector TeX/PDF report
    build_tex_and_pdf(v2_rows, eval_v2, eval_v1, val_curves, test_curves, val_cm, test_cm)

    print("[OK] Generated TensorBoard events:", tb_v2.name, tb_v1.name)
    print("[OK] Dashboard:", V2_TRAIN_DIR / "tensorboard_dashboard.html")
    print("[OK] Vector PDF:", V2_TRAIN_DIR / "roc_f1_tensorboard_summary.pdf")
    print("[OK] Epoch CSV:", epoch_csv_path)


if __name__ == "__main__":
    main()
