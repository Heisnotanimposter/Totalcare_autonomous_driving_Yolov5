#!/usr/bin/env python3
"""
KITTI Multi-Object Tracking Inspection, Audit & Sequence-Split YOLO Preparation.

Static-audit remediation update (PHASE 0 — experimental tooling, NOT FINAL):
  1. Track IDs are sequence-scoped: always keyed by (sequence_id, track_id).
  2. Sequence-level disjoint Train/Val/Test splits only (never random frame splitting).
  3. Decoupled pixel-mask policies (`raw`, `dontcare_only`, `distractor_only`,
     `dontcare_and_distractors`) with explicit `Van`, `Person`, and `Person_sitting` handling.
  4. Full image/label correspondence counts (missing, extra, duplicate, unreadable) and
     per-split class coverage validation before training.
"""

import argparse
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

IMAGE_ENTRY_RE = re.compile(r"(?:training/)?image_02/(\d{4})/(\d+)\.(png|jpg)$")
LABEL_ENTRY_RE = re.compile(r"(?:training/)?label_02/(\d{4})\.txt$")

KNOWN_KITTI_CLASSES = {
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
}


def validate_disjoint_sequences(train_seqs: Iterable[str], val_seqs: Iterable[str], test_seqs: Iterable[str]) -> None:
    """Verify train, validation, and test splits share zero sequence IDs."""
    sets = {"train": set(train_seqs), "val": set(val_seqs), "test": set(test_seqs)}
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise ValueError(f"Sequence split leakage between {names[i]} and {names[j]}: {sorted(overlap)}")


def validate_target_classes(target_classes: Sequence[str]) -> List[str]:
    """Validate target KITTI class names for emptiness, duplicates, and known taxonomy."""
    classes = list(target_classes)
    if not classes:
        raise ValueError("target_classes must contain at least one class name.")
    if len(set(classes)) != len(classes):
        raise ValueError(f"Duplicate class names in target_classes: {classes}")
    if "DontCare" in classes:
        raise ValueError("'DontCare' is an ignore region and cannot be used as a trainable target class.")
    unknown = [c for c in classes if c not in KNOWN_KITTI_CLASSES]
    if unknown:
        raise ValueError(f"Unknown KITTI class names {unknown}; allowed: {sorted(KNOWN_KITTI_CLASSES)}")
    return classes


def resolve_distractor_policy(
    target_classes: Sequence[str],
    mask_dontcare: bool,
    mask_distractors: bool,
    extra_distractors: Optional[Sequence[str]] = None,
) -> Tuple[List[str], str]:
    """
    Determine distractor classes to ignore/mask and return (distractors, policy_mode).
    Note: official TrackEval ignores Van for Car and Person/Person_sitting for Pedestrian.
    """
    cls_set = set(target_classes)
    distractors = set()
    if mask_distractors:
        if "Car" in cls_set and "Van" not in cls_set:
            distractors.add("Van")
        if "Pedestrian" in cls_set:
            if "Person" not in cls_set:
                distractors.add("Person")
            if "Person_sitting" not in cls_set:
                distractors.add("Person_sitting")
        if extra_distractors:
            for cls_name in extra_distractors:
                if cls_name not in cls_set and cls_name != "DontCare":
                    distractors.add(cls_name)

    if mask_dontcare and mask_distractors:
        policy = "dontcare_and_distractors"
    elif mask_dontcare:
        policy = "dontcare_only"
    elif mask_distractors:
        policy = "distractor_only"
    else:
        policy = "raw"
    return sorted(distractors), policy


def parse_kitti_label_lines(lines: Iterable[str], seq_id: str) -> Dict:
    """
    Parse one KITTI label_02/XXXX.txt file into per-frame objects and sequence-scoped tracklets.
    Returns dict with objects_by_frame, dontcare_by_frame, tracklet stats, and labeled frame indices.
    """
    objects_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    dontcare_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    tracklet_classes: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    class_counts = Counter()
    frames_seen = set()

    for raw in lines:
        parts = raw.strip().split()
        if not parts:
            continue
        frame_idx = int(parts[0])
        track_id = int(parts[1])
        cls_name = parts[2]
        x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])

        frames_seen.add(frame_idx)
        class_counts[cls_name] += 1

        if cls_name == "DontCare" or track_id < 0:
            dontcare_by_frame[frame_idx].append((x1, y1, x2, y2))
            continue

        tracklet_classes[(seq_id, track_id)][cls_name] += 1
        objects_by_frame[frame_idx].append({
            "frame": frame_idx,
            "track_id": track_id,
            "class_name": cls_name,
            "truncated": float(parts[3]),
            "occluded": int(parts[4]),
            "bbox": (x1, y1, x2, y2),
        })

    return {
        "seq_id": seq_id,
        "max_frame": max(frames_seen) if frames_seen else -1,
        "labeled_frames": len(frames_seen),
        "frames_seen": sorted(frames_seen),
        "objects_by_frame": dict(objects_by_frame),
        "dontcare_by_frame": dict(dontcare_by_frame),
        "sequence_scoped_tracklets": len(tracklet_classes),
        "class_counts": dict(class_counts),
    }


def load_labels_from_zip(label_zip: Path) -> Dict[str, Dict]:
    """Read all training/label_02/*.txt sequences from data_tracking_label_2.zip."""
    parsed: Dict[str, Dict] = {}
    with zipfile.ZipFile(label_zip, "r") as zf:
        for entry in sorted(zf.namelist()):
            m = LABEL_ENTRY_RE.match(entry)
            if not m:
                continue
            seq_id = m.group(1)
            lines = zf.read(entry).decode("utf-8").splitlines()
            parsed[seq_id] = parse_kitti_label_lines(lines, seq_id)
    return parsed


class _KittiImageSource:
    """Unified reader for KITTI training/image_02 frames from either a ZIP archive or directory."""

    def __init__(self, image_zip: Optional[Path] = None, image_dir: Optional[Path] = None):
        if (image_zip is None) == (image_dir is None):
            raise ValueError("Provide exactly one of image_zip or image_dir.")
        self.image_zip = Path(image_zip) if image_zip is not None else None
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self._zf: Optional[zipfile.ZipFile] = None
        self.img_index: Dict[Tuple[str, int], object] = {}
        self.duplicate_entries: Dict[str, List[str]] = defaultdict(list)

    def __enter__(self) -> "_KittiImageSource":
        if self.image_zip is not None:
            self._zf = zipfile.ZipFile(self.image_zip, "r")
            entries = sorted(self._zf.namelist())
            for entry in entries:
                m = IMAGE_ENTRY_RE.match(entry)
                if not m:
                    continue
                self._register(m.group(1), int(m.group(2)), entry, entry)
        else:
            assert self.image_dir is not None
            root = self.image_dir
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".png", ".jpg"}:
                    continue
                rel = path.relative_to(root).as_posix()
                m = IMAGE_ENTRY_RE.match(rel)
                if not m:
                    # Also allow pointing directly at .../image_02/<seq>/<frame>.png
                    m_direct = re.match(r"^(\d{4})/(\d+)\.(png|jpg)$", rel)
                    if not m_direct:
                        continue
                    seq_id, fr_str = m_direct.group(1), m_direct.group(2)
                else:
                    seq_id, fr_str = m.group(1), m.group(2)
                self._register(seq_id, int(fr_str), path, rel)
        return self

    def _register(self, seq_id: str, fr_idx: int, handle: object, label: str) -> None:
        key = (seq_id, fr_idx)
        if key in self.img_index:
            self.duplicate_entries[seq_id].append(label)
            return
        self.img_index[key] = handle

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def test_integrity(self) -> Optional[str]:
        return self._zf.testzip() if self._zf is not None else None

    def read_bytes(self, key: Tuple[str, int]) -> bytes:
        handle = self.img_index[key]
        if self._zf is not None:
            return self._zf.read(str(handle))
        return Path(handle).read_bytes()  # type: ignore[arg-type]


def _inspect_image_bytes(raw: bytes, label: str, full_verify: bool) -> Tuple[Optional[Tuple[int, int, str]], Optional[str]]:
    """Decode raw image bytes and optionally verify full byte stream decode."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            size_mode = (im.size[0], im.size[1], im.mode)
            if full_verify:
                im.verify()
        return size_mode, None
    except Exception as exc:
        return None, f"{label}: {exc}"


def audit_images_and_labels(
    image_zip: Optional[Path],
    label_zip: Path,
    expected_total_frames: int = 8008,
    verify_all_images: bool = True,
    image_dir: Optional[Path] = None,
) -> Dict:
    """
    Phase-1 / Step-2 verification across KITTI image_2 (ZIP or directory) and label_2 ZIP.
    Checks archive integrity, duplicate/unreadable frames, full missing/extra counts, and resolutions.
    """
    labels = load_labels_from_zip(label_zip)
    img_frames: Dict[str, Dict[int, object]] = defaultdict(dict)
    unreadable_entries: List[str] = []
    resolutions = Counter()

    with _KittiImageSource(image_zip=image_zip, image_dir=image_dir) as src:
        bad_member = src.test_integrity()
        duplicate_entries = dict(src.duplicate_entries)
        for (seq_id, fr_idx), handle in src.img_index.items():
            img_frames[seq_id][fr_idx] = handle

        for seq_id, fdict in sorted(img_frames.items()):
            ordered = sorted(fdict.keys())
            keys_to_check = ordered if verify_all_images else sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]})
            for fk in keys_to_check:
                raw = src.read_bytes((seq_id, fk))
                size_mode, err = _inspect_image_bytes(raw, str(fdict[fk]), full_verify=verify_all_images)
                if err is not None:
                    unreadable_entries.append(err)
                elif size_mode is not None:
                    resolutions[size_mode] += 1

    per_seq = {}
    totals = Counter()
    for seq_id in sorted(set(labels.keys()) | set(img_frames.keys())):
        actual_set = set(img_frames.get(seq_id, {}).keys())
        lbl = labels.get(seq_id, {})
        max_f = lbl.get("max_frame", -1)
        expected_span_set = set(range(max_f + 1)) if max_f >= 0 else set()
        labeled_frames_set = set(lbl.get("frames_seen", []))

        missing_in_span = sorted(expected_span_set - actual_set)
        extra_beyond_span = sorted(actual_set - expected_span_set)
        label_without_img = sorted(labeled_frames_set - actual_set)
        dup_cnt = len(duplicate_entries.get(seq_id, []))

        totals["total_training_images"] += len(actual_set)
        totals["missing_image_count"] += len(missing_in_span)
        totals["extra_image_count"] += len(extra_beyond_span)
        totals["duplicate_image_frame_count"] += dup_cnt
        totals["label_frame_without_image_count"] += len(label_without_img)

        per_seq[seq_id] = {
            "image_count": len(actual_set),
            "label_span_len": max_f + 1 if max_f >= 0 else 0,
            "labeled_frames": lbl.get("labeled_frames", 0),
            "missing_image_count": len(missing_in_span),
            "extra_image_count": len(extra_beyond_span),
            "duplicate_image_frame_count": dup_cnt,
            "label_frame_without_image_count": len(label_without_img),
            "missing_images_in_span": missing_in_span[:10],
            "extra_images_beyond_label_max": extra_beyond_span[:10],
        }

    return {
        "image_zip_corrupt_member": bad_member,
        "total_training_images": totals["total_training_images"],
        "expected_seqmap_frames": expected_total_frames,
        "missing_image_count": totals["missing_image_count"],
        "extra_image_count": totals["extra_image_count"],
        "duplicate_image_frame_count": totals["duplicate_image_frame_count"],
        "unreadable_image_count": len(unreadable_entries),
        "label_frame_without_image_count": totals["label_frame_without_image_count"],
        "unreadable_entries_sample": unreadable_entries[:10],
        "verify_all_images": verify_all_images,
        "sampled_resolutions": {f"{w}x{h}_{mode}": cnt for (w, h, mode), cnt in resolutions.items()},
        "per_sequence": per_seq,
    }


def mask_boxes_on_image(img_arr: np.ndarray, boxes: Sequence[Tuple[float, float, float, float]], fill_value: int = 114) -> np.ndarray:
    """Fill DontCare or ignored distractor regions with neutral pixel value (114 for YOLO padding)."""
    if not boxes:
        return img_arr
    out = img_arr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(w, int(round(x1))))
        iy1 = max(0, min(h, int(round(y1))))
        ix2 = max(0, min(w, int(round(x2))))
        iy2 = max(0, min(h, int(round(y2))))
        if ix2 > ix1 and iy2 > iy1:
            out[iy1:iy2, ix1:ix2] = fill_value
    return out


def xyxy_to_yolo(bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """Convert pixel [x1, y1, x2, y2] to clipped normalized YOLO (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    return ((x1 + bw / 2.0) / img_w, (y1 + bh / 2.0) / img_h, bw / img_w, bh / img_h)


def export_kitti_yolo_dataset(
    image_zip: Optional[Path],
    label_zip: Path,
    output_dir: Path,
    target_classes: Sequence[str],
    train_seqs: Sequence[str],
    val_seqs: Sequence[str],
    test_seqs: Sequence[str],
    mask_dontcare: bool = True,
    mask_distractors: bool = True,
    extra_distractors: Optional[Sequence[str]] = None,
    save_format: str = "jpg",
    strict_coverage: bool = False,
    image_dir: Optional[Path] = None,
    oversample_rare_classes: bool = False,
    cyclist_repeat: int = 4,
    pedestrian_repeat: int = 2,
) -> Dict:
    """
    Export sequence-split KITTI Tracking images and normalized YOLO labels.
    Applies `mask_dontcare` and `mask_distractors` independently (4 possible policies:
    raw, dontcare_only, distractor_only, dontcare_and_distractors).
    """
    validated_classes = validate_target_classes(target_classes)
    validate_disjoint_sequences(train_seqs, val_seqs, test_seqs)
    labels = load_labels_from_zip(label_zip)

    seq_to_split: Dict[str, str] = {}
    for split_name, seqs in (("train", train_seqs), ("val", val_seqs), ("test", test_seqs)):
        for s in seqs:
            if s not in labels:
                raise KeyError(f"Sequence '{s}' not found in {label_zip}")
            seq_to_split[s] = split_name

    cls_to_idx = {c: i for i, c in enumerate(validated_classes)}
    distractors_list, pixel_mask_policy = resolve_distractor_policy(
        validated_classes,
        mask_dontcare=mask_dontcare,
        mask_distractors=mask_distractors,
        extra_distractors=extra_distractors,
    )
    distractor_names = set(distractors_list)

    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    train_image_entries: List[str] = []

    splits_meta = {
        s: {
            "sequences": [],
            "frames": 0,
            "frames_with_boxes": 0,
            "empty_frames": 0,
            "boxes_by_cls": {c: 0 for c in validated_classes},
            "sequences_covering_cls": {c: set() for c in validated_classes},
            "dontcare_regions_masked": 0,
            "distractor_regions_masked": 0,
        }
        for s in ("train", "val", "test")
    }

    with _KittiImageSource(image_zip=image_zip, image_dir=image_dir) as img_src:
        img_index = img_src.img_index
        if img_src.duplicate_entries:
            raise ValueError(f"Duplicate image entries found: {dict(img_src.duplicate_entries)}")

        source_desc = image_zip or image_dir
        for seq_id in seq_to_split:
            if not any(s == seq_id for (s, _) in img_index):
                raise ValueError(f"Requested sequence '{seq_id}' has 0 images in {source_desc}")

        for seq_id, split_name in sorted(seq_to_split.items()):
            seq_lbl = labels[seq_id]
            sdata = splits_meta[split_name]
            sdata["sequences"].append(seq_id)
            seq_frames = sorted(fr for (s, fr) in img_index if s == seq_id)
            img_out_dir = output_dir / "images" / split_name
            lbl_out_dir = output_dir / "labels" / split_name

            for fr in seq_frames:
                with Image.open(io.BytesIO(img_src.read_bytes((seq_id, fr)))) as im:
                    rgb = np.array(im.convert("RGB"))
                img_h, img_w = rgb.shape[:2]

                objs = seq_lbl["objects_by_frame"].get(fr, [])
                dc_boxes = list(seq_lbl["dontcare_by_frame"].get(fr, []))
                extra_mask_boxes = [o["bbox"] for o in objs if o["class_name"] in distractor_names]

                # Decoupled pixel masking: each option independently controls its own boxes
                boxes_to_mask: List[Tuple[float, float, float, float]] = []
                if mask_dontcare and dc_boxes:
                    boxes_to_mask.extend(dc_boxes)
                    sdata["dontcare_regions_masked"] += len(dc_boxes)
                if mask_distractors and extra_mask_boxes:
                    boxes_to_mask.extend(extra_mask_boxes)
                    sdata["distractor_regions_masked"] += len(extra_mask_boxes)
                if boxes_to_mask:
                    rgb = mask_boxes_on_image(rgb, boxes_to_mask, fill_value=114)

                yolo_lines = []
                for obj in objs:
                    cname = obj["class_name"]
                    if cname not in cls_to_idx:
                        continue
                    yolo_box = xyxy_to_yolo(obj["bbox"], img_w, img_h)
                    if yolo_box is None:
                        continue
                    cidx = cls_to_idx[cname]
                    yolo_lines.append(f"{cidx} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                    sdata["boxes_by_cls"][cname] += 1
                    sdata["sequences_covering_cls"][cname].add(seq_id)

                if yolo_lines:
                    sdata["frames_with_boxes"] += 1
                else:
                    sdata["empty_frames"] += 1

                stem = f"{seq_id}_{fr:06d}"
                img_out_path = img_out_dir / f"{stem}.{save_format}"
                Image.fromarray(rgb).save(img_out_path)
                with open(lbl_out_dir / f"{stem}.txt", "w", encoding="utf-8") as lf:
                    lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                sdata["frames"] += 1

                if split_name == "train":
                    present_cls = {o["class_name"] for o in objs if o["class_name"] in cls_to_idx}
                    rep = 1
                    if oversample_rare_classes:
                        if "Cyclist" in present_cls:
                            rep = max(rep, cyclist_repeat)
                        if "Pedestrian" in present_cls:
                            rep = max(rep, pedestrian_repeat)
                    abs_img = str(img_out_path.resolve())
                    for _ in range(rep):
                        train_image_entries.append(abs_img)

    coverage_warnings: List[str] = []
    formatted_splits = {}
    for s, d in splits_meta.items():
        missing_cls = [c for c in validated_classes if d["boxes_by_cls"][c] == 0]
        if d["frames"] > 0 and missing_cls:
            msg = f"Split '{s}' is missing target classes: {missing_cls}"
            coverage_warnings.append(msg)
            if strict_coverage:
                raise ValueError(msg)
        formatted_splits[s] = {
            "sequences": d["sequences"],
            "frames": d["frames"],
            "frames_with_boxes": d["frames_with_boxes"],
            "empty_frames": d["empty_frames"],
            "empty_frame_ratio": round(d["empty_frames"] / d["frames"], 4) if d["frames"] else 0.0,
            "boxes_by_cls": d["boxes_by_cls"],
            "sequence_count_by_cls": {c: len(v) for c, v in d["sequences_covering_cls"].items()},
            "sequences_covering_cls": {c: sorted(v) for c, v in d["sequences_covering_cls"].items()},
            "missing_classes": missing_cls,
            "dontcare_regions_masked": d["dontcare_regions_masked"],
            "distractor_regions_masked": d["distractor_regions_masked"],
        }

    train_entry = "images/train"
    if oversample_rare_classes and train_image_entries:
        train_list_path = output_dir / "train_weighted.txt"
        train_list_path.write_text("\n".join(train_image_entries) + "\n", encoding="utf-8")
        train_entry = "train_weighted.txt"

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": train_entry,
        "val": "images/val",
        "test": "images/test",
        "nc": len(validated_classes),
        "names": {i: name for i, name in enumerate(validated_classes)},
    }
    with open(output_dir / "data_kitti.yaml", "w", encoding="utf-8") as yf:
        yaml.safe_dump(data_yaml, yf, sort_keys=False)

    out_json = {
        "target_classes": validated_classes,
        "pixel_mask_policy": pixel_mask_policy,
        "mask_dontcare": mask_dontcare,
        "mask_distractors": mask_distractors,
        "masked_distractors": distractors_list,
        "oversample_rare_classes": oversample_rare_classes,
        "weighted_train_entries": len(train_image_entries) if oversample_rare_classes else formatted_splits["train"]["frames"],
        "coverage_warnings": coverage_warnings,
        "splits": formatted_splits,
    }
    with open(output_dir / "kitti_export_manifest.json", "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI Tracking audit and sequence-split YOLO preparer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Verify KITTI image_2 (ZIP or dir) + label_2 ZIP integrity and full frame correspondence")
    g_audit = p_audit.add_mutually_exclusive_group(required=True)
    g_audit.add_argument("--image-zip", type=Path)
    g_audit.add_argument("--image-dir", type=Path)
    p_audit.add_argument("--label-zip", type=Path, required=True)
    p_audit.add_argument("--sample-decode-only", action="store_true", help="Check 3 sample frames/seq instead of decoding every PNG")

    p_export = sub.add_parser("export", help="Export sequence-disjoint KITTI YOLO dataset")
    g_export = p_export.add_mutually_exclusive_group(required=True)
    g_export.add_argument("--image-zip", type=Path)
    g_export.add_argument("--image-dir", type=Path)
    p_export.add_argument("--label-zip", type=Path, required=True)
    p_export.add_argument("--output-dir", type=Path, required=True)
    p_export.add_argument("--classes", type=str, nargs="+", required=True, help="e.g. Car Pedestrian Cyclist")
    p_export.add_argument("--train-seqs", type=str, nargs="+", required=True)
    p_export.add_argument("--val-seqs", type=str, nargs="+", required=True)
    p_export.add_argument("--test-seqs", type=str, nargs="+", default=[])
    p_export.add_argument(
        "--pixel-mask-policy",
        type=str,
        choices=["raw", "dontcare_only", "distractor_only", "dontcare_and_distractors"],
        default=None,
        help="Explicitly set masking policy (overrides --no-mask-* flags when provided).",
    )
    p_export.add_argument("--no-mask-dontcare", action="store_true")
    p_export.add_argument("--no-mask-distractors", action="store_true")
    p_export.add_argument("--extra-distractors", type=str, nargs="*", default=None)
    p_export.add_argument("--strict-coverage", action="store_true", help="Fail if any non-empty split lacks a target class")
    p_export.add_argument("--oversample-rare-classes", action="store_true", help="Oversample Cyclist/Pedestrian frames in train split only")
    p_export.add_argument("--save-format", type=str, default="jpg", choices=["jpg", "png"])

    args = parser.parse_args()
    if args.command == "audit":
        res = audit_images_and_labels(
            args.image_zip,
            args.label_zip,
            verify_all_images=not args.sample_decode_only,
            image_dir=args.image_dir,
        )
        print(json.dumps(res, indent=2))
    elif args.command == "export":
        if args.pixel_mask_policy == "raw":
            m_dc, m_dist = False, False
        elif args.pixel_mask_policy == "dontcare_only":
            m_dc, m_dist = True, False
        elif args.pixel_mask_policy == "distractor_only":
            m_dc, m_dist = False, True
        elif args.pixel_mask_policy == "dontcare_and_distractors":
            m_dc, m_dist = True, True
        else:
            m_dc = not args.no_mask_dontcare
            m_dist = not args.no_mask_distractors

        res = export_kitti_yolo_dataset(
            image_zip=args.image_zip,
            label_zip=args.label_zip,
            output_dir=args.output_dir,
            target_classes=args.classes,
            train_seqs=args.train_seqs,
            val_seqs=args.val_seqs,
            test_seqs=args.test_seqs,
            mask_dontcare=m_dc,
            mask_distractors=m_dist,
            extra_distractors=args.extra_distractors,
            save_format=args.save_format,
            strict_coverage=args.strict_coverage,
            image_dir=args.image_dir,
            oversample_rare_classes=args.oversample_rare_classes,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
