#!/usr/bin/env python3
"""Kaggle v4 GPU Kernel: Deduped KITTI-Det + KITTI-Track YOLO11l @ 1280 Training, TTA/Difficulty Eval, ByteTrack + MobileSAM."""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

KITTI_PREP_B64 = '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'
TRACK_UTILS_B64 = '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'

TARGET_CLASSES = ["Car", "Pedestrian", "Cyclist"]
TRAIN_SEQS = ["0000", "0001", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0014", "0018", "0020"]
VAL_SEQS = ["0002", "0012", "0015", "0017"]
TEST_SEQS = ["0013", "0016", "0019"]
DEMO_SEQS = ["0015", "0019"]

S3_LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip"
S3_CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip"
S3_OXTS_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_oxts.zip"


def run(cmd, check=True):
    print(f"[CMD] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, check=check)


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    curl = shutil.which("curl")
    if curl:
        run([curl, "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", str(dst), url])
    else:
        with urllib.request.urlopen(url) as resp, open(dst, "wb") as out:
            shutil.copyfileobj(resp, out)


def compute_dhash64(img_path: Path) -> int:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(bool(b))
    return val


def file_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def find_kaggle_tracking_dir() -> Path:
    for cand in [
        Path("/kaggle/input/datasets/leducnhuan/kitti-tracking/kitti_tracking/training/image_02"),
        Path("/kaggle/input/leducnhuan/kitti-tracking/kitti_tracking/training/image_02"),
        Path("/kaggle/input/leducnhuan/kitti-tracking/training/image_02"),
    ]:
        if cand.is_dir():
            return cand
    for p in Path("/kaggle/input").rglob("image_02"):
        if p.is_dir() and (p / "0015").is_dir() and (p / "0019").is_dir():
            return p
    raise FileNotFoundError("KITTI Tracking image_02 not found in /kaggle/input")


def find_kitti_det_root() -> Optional[Path]:
    for cand in [
        Path("/kaggle/input/datasets/harshpdani/kitti-stereo-image/kitti"),
        Path("/kaggle/input/harshpdani/kitti-stereo-image/kitti"),
    ]:
        if (cand / "images").is_dir() and (cand / "labels").is_dir():
            return cand
    for p in Path("/kaggle/input").rglob("classes.json"):
        if (p.parent / "images").is_dir() and (p.parent / "labels").is_dir():
            return p.parent
    return None


def compute_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    aa = np.maximum(0.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    bb = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(1e-9, aa + bb - inter)


def ap_101(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def eval_kitti_difficulties(
    model,
    split_name: str,
    seqs: Sequence[str],
    image_dir: Path,
    label_root: Path,
    imgsz: int = 1280,
    augment: bool = True,
) -> Dict:
    from kitti_prep import parse_kitti_label_lines

    diff_rules = {
        "Easy": lambda h, occ, tr: (h >= 40.0 and occ == 0 and tr <= 0.15),
        "Moderate": lambda h, occ, tr: (h >= 25.0 and occ <= 1 and tr <= 0.30),
        "Hard": lambda h, occ, tr: (h >= 25.0 and occ <= 2 and tr <= 0.50),
        "All": lambda h, occ, tr: True,
    }

    gt_by_frame = {}
    dontcare_by_frame = {}
    img_list = []
    for seq in seqs:
        lbl_p = label_root / "training" / "label_02" / f"{seq}.txt"
        parsed = parse_kitti_label_lines(lbl_p.read_text(encoding="utf-8").splitlines(), seq)
        for img_p in sorted((image_dir / seq).glob("*.png")):
            fr = int(img_p.stem)
            key = f"{seq}_{fr:06d}"
            img_list.append((key, img_p))
            gt_by_frame[key] = parsed["objects_by_frame"].get(fr, [])
            dontcare_by_frame[key] = parsed["dontcare_by_frame"].get(fr, [])

    preds_by_key = {}
    bs = 8
    for i in range(0, len(img_list), bs):
        chunk = img_list[i : i + bs]
        res_list = model.predict(
            [str(p) for _, p in chunk],
            imgsz=imgsz,
            conf=0.001,
            iou=0.65,
            augment=augment,
            device=0,
            verbose=False,
        )
        for (key, _), res in zip(chunk, res_list):
            if res.boxes is None or len(res.boxes) == 0:
                preds_by_key[key] = []
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            preds_by_key[key] = [(float(c), TARGET_CLASSES[k], b.tolist()) for b, c, k in zip(xyxy, confs, clss) if k < len(TARGET_CLASSES)]

    # KITTI-official style evaluation per difficulty level (with DontCare + out-of-difficulty GT ignored)
    out = {}
    for dname, rule in diff_rules.items():
        per_cls_ap = {}
        per_cls_best_f1 = {}
        for cls in TARGET_CLASSES:
            all_preds = []  # (conf, tp, fp)
            total_gt = 0
            for key, _ in img_list:
                raw_objs = gt_by_frame[key]
                dc_boxes = dontcare_by_frame[key]
                keep_gt = []
                ign_gt = list(dc_boxes)
                for o in raw_objs:
                    if o["class_name"] == cls:
                        x1, y1, x2, y2 = o["bbox"]
                        if rule(y2 - y1, o["occluded"], o["truncated"]):
                            keep_gt.append([x1, y1, x2, y2])
                        else:
                            ign_gt.append((x1, y1, x2, y2))
                    elif (cls == "Car" and o["class_name"] == "Van") or (cls == "Pedestrian" and o["class_name"] in ("Person", "Person_sitting")):
                        ign_gt.append(o["bbox"])
                total_gt += len(keep_gt)
                matched = [False] * len(keep_gt)

                cpreds = sorted([p for p in preds_by_key[key] if p[1] == cls], key=lambda t: -t[0])
                if not cpreds:
                    continue
                gt_arr = np.array(keep_gt, dtype=np.float32) if keep_gt else None
                ign_arr = np.array(ign_gt, dtype=np.float32) if ign_gt else None
                pred_arr = np.array([p[2] for p in cpreds], dtype=np.float32)
                ious_gt = compute_iou_matrix(pred_arr, gt_arr) if gt_arr is not None else None
                ious_ign = compute_iou_matrix(pred_arr, ign_arr) if ign_arr is not None else None

                for pi, (cf, _, pb) in enumerate(cpreds):
                    if ious_gt is not None and len(keep_gt) > 0:
                        best_gi = int(np.argmax(ious_gt[pi]))
                        best_iou = float(ious_gt[pi, best_gi])
                        if best_iou >= 0.5 and not matched[best_gi]:
                            matched[best_gi] = True
                            all_preds.append((cf, 1, 0))
                            continue
                    # Ignore if overlapping ignored region (DontCare or non-difficulty GT or distractor) >= 0.5 IoA/IoU
                    if ious_ign is not None and len(ign_gt) > 0:
                        px1, py1, px2, py2 = pb
                        p_area = max(1e-6, (px2 - px1) * (py2 - py1))
                        ignore_hit = False
                        for igb in ign_gt:
                            ix1, iy1, ix2, iy2 = max(px1, igb[0]), max(py1, igb[1]), min(px2, igb[2]), min(py2, igb[3])
                            if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1) / p_area) >= 0.5:
                                ignore_hit = True
                                break
                        if ignore_hit:
                            continue
                    all_preds.append((cf, 0, 1))

            if total_gt == 0 or not all_preds:
                per_cls_ap[cls] = 0.0
                per_cls_best_f1[cls] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "conf": 0.0}
                continue
            all_preds.sort(key=lambda t: -t[0])
            tps = np.cumsum([t[1] for t in all_preds], dtype=np.float64)
            fps = np.cumsum([t[2] for t in all_preds], dtype=np.float64)
            confs_arr = np.array([t[0] for t in all_preds], dtype=np.float64)
            rec = tps / max(1, total_gt)
            prec = tps / np.maximum(1e-9, tps + fps)
            f1s = 2.0 * prec * rec / np.maximum(1e-9, prec + rec)
            best_i = int(np.argmax(f1s))
            per_cls_ap[cls] = round(ap_101(rec, prec), 4)
            per_cls_best_f1[cls] = {
                "f1": round(float(f1s[best_i]), 4),
                "precision": round(float(prec[best_i]), 4),
                "recall": round(float(rec[best_i]), 4),
                "conf": round(float(confs_arr[best_i]), 3),
            }
        out[dname] = {
            "map50": round(float(np.mean(list(per_cls_ap.values()))), 4),
            "mean_peak_f1": round(float(np.mean([v["f1"] for v in per_cls_best_f1.values()])), 4),
            "mean_precision_at_peak_f1": round(float(np.mean([v["precision"] for v in per_cls_best_f1.values()])), 4),
            "mean_recall_at_peak_f1": round(float(np.mean([v["recall"] for v in per_cls_best_f1.values()])), 4),
            "per_class_ap50": per_cls_ap,
            "per_class_peak_f1": per_cls_best_f1,
        }
    return out


def main() -> None:
    work_dir = Path("/kaggle/working")
    out_dir = work_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/tmp/totalcare_v4")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.0.0", "lap>=0.5.12", "pyyaml"])
    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "kitti_prep.py").write_bytes(zlib.decompress(base64.b64decode(KITTI_PREP_B64)))
    (src_dir / "track_utils.py").write_bytes(zlib.decompress(base64.b64decode(TRACK_UTILS_B64)))
    sys.path.insert(0, str(src_dir))

    import torch
    from ultralytics import YOLO
    from kitti_prep import export_kitti_yolo_dataset, mask_boxes_on_image
    from track_utils import run_traffic_tracking

    label_zip = tmp_dir / "data_tracking_label_2.zip"
    calib_zip = tmp_dir / "data_tracking_calib.zip"
    oxts_zip = tmp_dir / "data_tracking_oxts.zip"
    download_file(S3_LABEL_URL, label_zip)
    download_file(S3_CALIB_URL, calib_zip)
    download_file(S3_OXTS_URL, oxts_zip)

    with zipfile.ZipFile(label_zip, "r") as zf:
        zf.extractall(tmp_dir / "extracted_labels")

    image_dir = find_kaggle_tracking_dir()
    yolo_dir = tmp_dir / "kitti_yolo_v4"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    manifest = export_kitti_yolo_dataset(
        image_zip=None,
        label_zip=label_zip,
        output_dir=yolo_dir,
        target_classes=TARGET_CLASSES,
        train_seqs=TRAIN_SEQS,
        val_seqs=VAL_SEQS,
        test_seqs=TEST_SEQS,
        mask_dontcare=True,
        mask_distractors=True,
        save_format="jpg",
        strict_coverage=False,
        image_dir=image_dir,
        oversample_rare_classes=True,
        cyclist_repeat=3,
        pedestrian_repeat=2,
    )

    # Deduplicate & append KITTI 2D Object Detection (harshpdani/kitti-stereo-image) to Train ONLY
    heldout_md5 = set()
    heldout_dhash = []
    for seq in VAL_SEQS + TEST_SEQS:
        for p in (image_dir / seq).glob("*.png"):
            heldout_md5.add(file_md5(p))
            heldout_dhash.append(compute_dhash64(p))
    heldout_dhash_arr = np.array(heldout_dhash, dtype=np.uint64)

    det_root = find_kitti_det_root()
    det_stats = {"found": det_root is not None, "total_scanned": 0, "excluded_leak": 0, "added_unique": 0, "boxes_added": Counter()}
    extra_train_entries = []
    if det_root is not None:
        # Map 8-class harshpdani IDs -> our 3 classes (0=Car, 1=Ped, 2=Cyc), mask Van(2) & Person_sitting(7)
        id_map = {0: 0, 1: 1, 3: 2}
        mask_ids = {2, 7}
        for sub in ("train", "val"):
            img_sub = det_root / "images" / sub
            lbl_sub = det_root / "labels" / sub
            if not img_sub.is_dir():
                continue
            for img_p in sorted(img_sub.glob("*.png")):
                det_stats["total_scanned"] += 1
                if file_md5(img_p) in heldout_md5:
                    det_stats["excluded_leak"] += 1
                    continue
                dh = compute_dhash64(img_p)
                if len(heldout_dhash_arr) > 0:
                    xor = heldout_dhash_arr ^ np.uint64(dh)
                    # fast byte popcount check via bit_count on top candidate
                    dists = [int((int(x)).bit_count()) for x in xor[::4]]
                    if min(dists) <= 2:
                        det_stats["excluded_leak"] += 1
                        continue

                lbl_p = lbl_sub / f"{img_p.stem}.txt"
                if not lbl_p.is_file():
                    continue
                rgb = np.array(Image.open(img_p).convert("RGB"))
                ih, iw = rgb.shape[:2]
                keep_lines = []
                mask_xyxy = []
                has_cyc = False
                has_ped = False
                for line in lbl_p.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:])
                    if cid in mask_ids:
                        x1 = (cx - bw / 2.0) * iw
                        y1 = (cy - bh / 2.0) * ih
                        x2 = (cx + bw / 2.0) * iw
                        y2 = (cy + bh / 2.0) * ih
                        mask_xyxy.append((x1, y1, x2, y2))
                    elif cid in id_map:
                        nc = id_map[cid]
                        keep_lines.append(f"{nc} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        det_stats["boxes_added"][TARGET_CLASSES[nc]] += 1
                        if nc == 2:
                            has_cyc = True
                        elif nc == 1:
                            has_ped = True
                if not keep_lines:
                    continue
                rgb = mask_boxes_on_image(rgb, mask_xyxy, fill_value=114)
                out_img = yolo_dir / "images" / "train" / f"kdet_{sub}_{img_p.stem}.jpg"
                out_lbl = yolo_dir / "labels" / "train" / f"kdet_{sub}_{img_p.stem}.txt"
                Image.fromarray(rgb).save(out_img, quality=95)
                out_lbl.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")
                det_stats["added_unique"] += 1
                rep = 2 if has_cyc else 1
                abs_p = str(out_img.resolve())
                for _ in range(rep):
                    extra_train_entries.append(abs_p)
    det_stats["boxes_added"] = dict(det_stats["boxes_added"])
    print("[DET-MERGE]", json.dumps(det_stats), flush=True)

    train_txt = yolo_dir / "train_weighted.txt"
    base_entries = train_txt.read_text(encoding="utf-8").splitlines() if train_txt.is_file() else []
    merged_entries = base_entries + extra_train_entries
    (yolo_dir / "train_v4.txt").write_text("\n".join(merged_entries) + "\n", encoding="utf-8")

    data_yaml = yolo_dir / "data_kitti.yaml"
    yaml_txt = data_yaml.read_text(encoding="utf-8").replace("train_weighted.txt", "train_v4.txt")
    data_yaml.write_text(yaml_txt, encoding="utf-8")

    # Train YOLO11l @ imgsz=1280
    runs_dir = tmp_dir / "runs"
    run_name = "kitti_v4_yolo11l_1280"
    model = YOLO("yolo11l.pt")
    gpu_count = torch.cuda.device_count()
    train_device = list(range(gpu_count)) if gpu_count > 1 else 0
    print(f"[TRAIN] yolo11l.pt @ 1280px, entries={len(merged_entries)}, gpus={gpu_count}", flush=True)
    model.train(
        data=str(data_yaml),
        epochs=25,
        imgsz=1280,
        batch=8 if gpu_count > 1 else 4,
        nbs=64,
        device=train_device,
        workers=4,
        patience=9,
        cos_lr=True,
        close_mosaic=7,
        mixup=0.15,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    exp_dir = runs_dir / run_name
    best_weights = exp_dir / "weights" / "best.pt"
    assert best_weights.is_file()
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, out_dir / "weights" / "best.pt")

    t_out = out_dir / "train_results"
    t_out.mkdir(parents=True, exist_ok=True)
    for fn in ["results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "args.yaml"]:
        if (exp_dir / fn).is_file():
            shutil.copy2(exp_dir / fn, t_out / fn)

    eval_model = YOLO(str(best_weights))
    # Standard single-scale + Multi-scale TTA Ultralytics Val/Test + curve export
    val_std = eval_model.val(data=str(data_yaml), split="val", imgsz=1280, device=0, project=str(out_dir / "runs"), name="val_std", exist_ok=True)
    val_tta = eval_model.val(data=str(data_yaml), split="val", imgsz=1280, augment=True, device=0, project=str(out_dir / "runs"), name="val_tta", exist_ok=True)
    test_tta = eval_model.val(data=str(data_yaml), split="test", imgsz=1280, augment=True, device=0, project=str(out_dir / "runs"), name="test_tta", exist_ok=True)

    for fn in ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"]:
        if (out_dir / "runs" / "val_tta" / fn).is_file():
            shutil.copy2(out_dir / "runs" / "val_tta" / fn, t_out / fn)
        if (out_dir / "runs" / "test_tta" / fn).is_file():
            shutil.copy2(out_dir / "runs" / "test_tta" / fn, t_out / f"test_{fn}")

    def _sum_m(m):
        return {
            "map50": round(float(m.box.map50), 4),
            "map50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "per_class_ap50": {TARGET_CLASSES[i]: round(float(m.box.ap50[i]), 4) for i in range(len(TARGET_CLASSES)) if i < len(m.box.ap50)},
        }

    diff_val = eval_kitti_difficulties(eval_model, "val", VAL_SEQS, image_dir, tmp_dir / "extracted_labels", imgsz=1280, augment=True)
    diff_test = eval_kitti_difficulties(eval_model, "test", TEST_SEQS, image_dir, tmp_dir / "extracted_labels", imgsz=1280, augment=True)

    eval_summary = {
        "version": "v4_road_lane",
        "model_arch": "yolo11l.pt",
        "imgsz": 1280,
        "train_entries_total": len(merged_entries),
        "kitti_det_merge": det_stats,
        "val_std": _sum_m(val_std),
        "val_tta": _sum_m(val_tta),
        "test_tta": _sum_m(test_tta),
        "kitti_difficulty_val_tta": diff_val,
        "kitti_difficulty_test_tta": diff_test,
    }
    (out_dir / "eval_metrics_v4.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print("[EVAL v4]", json.dumps(eval_summary, indent=2), flush=True)

    # Track + MobileSAM on 0015 & 0019 using v4 detector
    track_summaries = {}
    for seq in DEMO_SEQS:
        d_raw = tmp_dir / f"raw_{seq}"
        d_raw.mkdir(parents=True, exist_ok=True)
        for p in sorted((image_dir / seq).glob("*.png")):
            shutil.copy2(p, d_raw / p.name)
        tsum = run_traffic_tracking(
            weights=best_weights,
            source=d_raw,
            output_dir=out_dir / f"track_{seq}",
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=1280,
            mask_model="mobile_sam.pt",
            fps_override=10.0,
            kitti_seq_name=seq,
            require_contiguous=True,
        )
        track_summaries[seq] = {
            "frames_processed": tsum["frames_processed"],
            "unique_track_ids_in_clip": tsum["unique_track_ids_in_clip"],
            "masked_instances_count": tsum.get("masked_instances_count"),
        }
    (out_dir / "tracking_summary_v4.json").write_text(json.dumps(track_summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
