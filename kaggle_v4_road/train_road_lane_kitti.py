#!/usr/bin/env python3
"""Kaggle GPU Kernel: KITTI v4 Road vs Non-Road Classification + Ego-Lane Segmentation + Inference."""

import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROAD_LANE_B64 = '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'

WORK = Path('/kaggle/working')
OUT_DIR = WORK / 'road_lane_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = WORK / 'src'
SRC_DIR.mkdir(parents=True, exist_ok=True)
(SRC_DIR / 'road_lane.py').write_bytes(zlib.decompress(base64.b64decode(ROAD_LANE_B64)))
sys.path.insert(0, str(SRC_DIR))

from road_lane import P2RoadMapper, TemporalRoadLaneProcessor, parse_kitti_road_gt_mask


def ensure_transformers():
    try:
        import transformers  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.38.0'], check=True)


def find_kitti_road_pairs() -> List[Dict]:
    roots = [Path('/kaggle/input'), Path('/kaggle/working')]
    img_map: Dict[str, Path] = {}
    road_gt_map: Dict[str, Path] = {}
    lane_gt_map: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.png'):
            parts = p.as_posix()
            if 'training/image_2/' in parts and re.match(r'^(um|umm|uu)_\d{6}\.png$', p.name):
                cat, idx = p.stem.split('_', 1)
                img_map[f'{cat}_{idx}'] = p
            elif 'training/gt_image_2/' in parts:
                m_r = re.match(r'^(um|umm|uu)_road_(\d{6})\.png$', p.name)
                m_l = re.match(r'^um_lane_(\d{6})\.png$', p.name)
                if m_r:
                    road_gt_map[f'{m_r.group(1)}_{m_r.group(2)}'] = p
                elif m_l:
                    lane_gt_map[f'um_{m_l.group(1)}'] = p
    pairs = []
    for k in sorted(img_map.keys()):
        if k in road_gt_map:
            pairs.append({
                'key': k,
                'cat': k.split('_', 1)[0],
                'img_path': img_map[k],
                'road_gt_path': road_gt_map[k],
                'lane_gt_path': lane_gt_map.get(k),
            })
    return pairs


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KittiRoadDataset(Dataset):
    def __init__(self, samples: List[Dict], height: int = 320, width: int = 1024, augment: bool = False):
        self.samples = samples
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        bgr = cv2.imread(str(item['img_path']))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        road_gt_bgr = cv2.imread(str(item['road_gt_path']))
        road_mask, _ = parse_kitti_road_gt_mask(road_gt_bgr, is_um_lane=False)
        has_lane = item['lane_gt_path'] is not None
        if has_lane:
            lane_gt_bgr = cv2.imread(str(item['lane_gt_path']))
            _, ego_mask = parse_kitti_road_gt_mask(lane_gt_bgr, is_um_lane=True)
            road_mask = np.maximum(road_mask, ego_mask)
        else:
            ego_mask = np.zeros_like(road_mask, dtype=np.uint8)

        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.fliplr(rgb).copy()
                road_mask = np.fliplr(road_mask).copy()
                ego_mask = np.fliplr(ego_mask).copy()
            if np.random.rand() < 0.6:
                alpha = float(np.random.uniform(0.82, 1.18))
                beta = float(np.random.uniform(-16.0, 16.0))
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        rgb_r = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        road_r = cv2.resize(road_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        ego_r = cv2.resize(ego_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # 3-class label: 0=Non-Road, 1=Other Road, 2=Ego-Lane
        cls3 = np.where(ego_r > 0, 2, np.where(road_r > 0, 1, 0)).astype(np.int64)

        x = (rgb_r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(road_r.astype(np.float32)),
            torch.from_numpy(cls3),
            torch.tensor(1 if has_lane else 0, dtype=torch.bool),
        )


def build_warm_segformer(device: torch.device) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation

    ckpt = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'
    print(f'[LOAD] Pretrained SegFormer: {ckpt}', flush=True)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt)

    old_head = model.decode_head.classifier
    in_ch = old_head.in_channels
    new_head = nn.Conv2d(in_ch, 3, kernel_size=1)
    with torch.no_grad():
        old_w = old_head.weight.data  # [19, in_ch, 1, 1]
        old_b = old_head.bias.data
        # Cityscapes idx 0 = road; non-road average over idx 1..18
        non_road_w = old_w[1:].mean(dim=0)
        non_road_b = old_b[1:].mean()
        new_head.weight.data[0] = non_road_w
        new_head.bias.data[0] = non_road_b
        new_head.weight.data[1] = old_w[0]
        new_head.bias.data[1] = old_b[0]
        new_head.weight.data[2] = old_w[0]
        new_head.bias.data[2] = old_b[0] - 0.15
    model.decode_head.classifier = new_head
    model.config.num_labels = 3
    return model.to(device)


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def find_kitti_tracking_root() -> Optional[Path]:
    for base in [Path('/kaggle/input'), Path('/kaggle/working')]:
        if not base.exists():
            continue
        cands = sorted(base.rglob('image_02'))
        # Strictly prefer training/image_02 (0015 has 376 frames, 0019 has 1059 frames; testing/ has 701 and 404!)
        for cand in cands:
            if (
                cand.is_dir()
                and 'training' in cand.parts
                and (cand / '0015').is_dir()
                and (cand / '0019').is_dir()
            ):
                return cand.parent
    return None


def ensure_calib_zip() -> Optional[Path]:
    for cand in list(Path('/kaggle/input').rglob('data_tracking_calib.zip')) + list(Path('/kaggle/working').rglob('data_tracking_calib.zip')):
        if cand.is_file():
            return cand
    dst = Path('/tmp/kitti_v4_calib/data_tracking_calib.zip')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file():
        url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip'
        print(f'[DL] {url}', flush=True)
        urllib.request.urlretrieve(url, str(dst))
    return dst


def parse_p2_from_calib_bytes(raw_txt: str) -> Optional[Tuple[float, float, float, float]]:
    for line in raw_txt.splitlines():
        if line.startswith('P2:'):
            vals = [float(x) for x in line.split(':', 1)[1].strip().split()]
            if len(vals) == 12:
                return vals[0], vals[5], vals[2], vals[6]
    return None


def main():
    ensure_transformers()
    pairs = find_kitti_road_pairs()
    print(f'[DATA] Found KITTI data_road pairs: {len(pairs)} (um_lane={sum(1 for p in pairs if p["lane_gt_path"])})', flush=True)
    if len(pairs) < 50:
        raise RuntimeError(f'Expected >=50 KITTI road pairs, found {len(pairs)}')

    # Deterministic stratified 80/20 split by category (um, umm, uu)
    train_samples, val_samples = [], []
    by_cat: Dict[str, List[Dict]] = {'um': [], 'umm': [], 'uu': []}
    for p in pairs:
        by_cat.setdefault(p['cat'], []).append(p)
    for cat, items in by_cat.items():
        for i, it in enumerate(items):
            if i % 5 == 0:
                val_samples.append(it)
            else:
                train_samples.append(it)
    print(f'[SPLIT] Train={len(train_samples)} imgs, Val={len(val_samples)} imgs', flush=True)

    H, W = 320, 1024
    train_ds = KittiRoadDataset(train_samples, height=H, width=W, augment=True)
    val_ds = KittiRoadDataset(val_samples, height=H, width=W, augment=False)
    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_warm_segformer(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_f1 = -1.0
    best_ckpt = OUT_DIR / 'road_segformer_best.pt'
    epoch_rows: List[Dict] = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_batches = 0
        for x, gt_road, gt_3cls, has_lane in train_loader:
            x = x.to(device, non_blocking=True)
            gt_road = gt_road.to(device, non_blocking=True)
            gt_3cls = gt_3cls.to(device, non_blocking=True)
            has_lane = has_lane.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(pixel_values=x).logits
                logits = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                road_logit = torch.logsumexp(logits[:, 1:3], dim=1) - logits[:, 0]
                loss_road = 0.65 * F.binary_cross_entropy_with_logits(road_logit, gt_road) + 0.35 * soft_dice_loss(
                    torch.sigmoid(road_logit), gt_road
                )
                loss = loss_road
                if has_lane.any():
                    loss_lane = F.cross_entropy(logits[has_lane], gt_3cls[has_lane])
                    loss = loss_road + 0.45 * loss_lane
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        n_vb = 0
        tp_r = fp_r = fn_r = tn_r = 0
        tp_l = fp_l = fn_l = 0
        with torch.no_grad():
            for x, gt_road, gt_3cls, has_lane in val_loader:
                x = x.to(device, non_blocking=True)
                gt_road = gt_road.to(device, non_blocking=True)
                gt_3cls = gt_3cls.to(device, non_blocking=True)
                has_lane = has_lane.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model(pixel_values=x).logits
                    logits = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                    road_logit = torch.logsumexp(logits[:, 1:3], dim=1) - logits[:, 0]
                    v_loss = F.binary_cross_entropy_with_logits(road_logit, gt_road)
                val_loss += float(v_loss.item())
                n_vb += 1
                prob_r = torch.sigmoid(road_logit)
                pred_r = prob_r >= 0.5
                gt_rb = gt_road >= 0.5
                tp_r += int((pred_r & gt_rb).sum().item())
                fp_r += int((pred_r & ~gt_rb).sum().item())
                fn_r += int((~pred_r & gt_rb).sum().item())
                tn_r += int((~pred_r & ~gt_rb).sum().item())

                if has_lane.any():
                    pred_cls = logits[has_lane].argmax(dim=1)
                    gt_e = gt_3cls[has_lane] == 2
                    pr_e = pred_cls == 2
                    tp_l += int((pr_e & gt_e).sum().item())
                    fp_l += int((pr_e & ~gt_e).sum().item())
                    fn_l += int((~pr_e & gt_e).sum().item())

        acc = (tp_r + tn_r) / max(1, tp_r + tn_r + fp_r + fn_r)
        prec = tp_r / max(1, tp_r + fp_r)
        rec = tp_r / max(1, tp_r + fn_r)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        iou_r = tp_r / max(1, tp_r + fp_r + fn_r)
        iou_nr = tn_r / max(1, tn_r + fp_r + fn_r)
        miou = 0.5 * (iou_r + iou_nr)
        lane_f1 = (2 * tp_l / max(1, 2 * tp_l + fp_l + fn_l)) if tp_l > 0 else 0.0
        lane_iou = (tp_l / max(1, tp_l + fp_l + fn_l)) if tp_l > 0 else 0.0

        row = {
            'epoch': ep,
            'time_sec': round(time.time() - t_start, 2),
            'train_loss': round(tr_loss / max(1, n_batches), 5),
            'val_loss': round(val_loss / max(1, n_vb), 5),
            'val_road_acc': round(acc, 5),
            'val_road_precision': round(prec, 5),
            'val_road_recall': round(rec, 5),
            'val_road_f1': round(f1, 5),
            'val_road_iou': round(iou_r, 5),
            'val_nonroad_iou': round(iou_nr, 5),
            'val_miou': round(miou, 5),
            'val_ego_lane_f1': round(lane_f1, 5),
            'val_ego_lane_iou': round(lane_iou, 5),
        }
        epoch_rows.append(row)
        print(
            f'[EP {ep:02d}/{epochs}] loss={row["train_loss"]:.4f}/{row["val_loss"]:.4f} '
            f'RoadAcc={acc:.4f} RoadF1={f1:.4f} mIoU={miou:.4f} EgoLaneF1={lane_f1:.4f}',
            flush=True,
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_ckpt)

    with (OUT_DIR / 'road_train_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    # Full 201-bin exact ROC / PR / F1 curve sweep on best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    bins = 200
    pos_hist = np.zeros(bins + 1, dtype=np.int64)
    neg_hist = np.zeros(bins + 1, dtype=np.int64)
    tp_l = fp_l = fn_l = 0
    with torch.no_grad():
        for x, gt_road, gt_3cls, has_lane in val_loader:
            x = x.to(device, non_blocking=True)
            logits = F.interpolate(model(pixel_values=x).logits, size=(H, W), mode='bilinear', align_corners=False)
            road_logit = torch.logsumexp(logits[:, 1:3], dim=1) - logits[:, 0]
            probs = torch.sigmoid(road_logit).cpu().numpy()
            gt_b = (gt_road.numpy() >= 0.5)
            q_bins = np.clip((probs * bins).astype(np.int64), 0, bins)
            pos_hist += np.bincount(q_bins[gt_b], minlength=bins + 1)
            neg_hist += np.bincount(q_bins[~gt_b], minlength=bins + 1)

            if has_lane.any():
                pred_cls = logits[has_lane].argmax(dim=1).cpu().numpy()
                gt_e = (gt_3cls[has_lane].numpy() == 2)
                pr_e = (pred_cls == 2)
                tp_l += int((pr_e & gt_e).sum())
                fp_l += int((pr_e & ~gt_e).sum())
                fn_l += int((~pr_e & gt_e).sum())

    total_pos = int(pos_hist.sum())
    total_neg = int(neg_hist.sum())
    # Cumulative counts >= threshold k/bins
    cum_pos = np.cumsum(pos_hist[::-1])[::-1]
    cum_neg = np.cumsum(neg_hist[::-1])[::-1]
    roc_curve = []
    pr_f1_curve = []
    for k in range(bins + 1):
        thr = round(k / bins, 3)
        tp = int(cum_pos[k])
        fp = int(cum_neg[k])
        fn = total_pos - tp
        tpr = tp / max(1, total_pos)
        fpr = fp / max(1, total_neg)
        prec = tp / max(1, tp + fp)
        f1_v = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
        acc_v = (tp + (total_neg - fp)) / max(1, total_pos + total_neg)
        roc_curve.append({'threshold': thr, 'tpr': round(tpr, 5), 'fpr': round(fpr, 5)})
        if k % 2 == 0:
            pr_f1_curve.append({'threshold': thr, 'precision': round(prec, 5), 'recall': round(tpr, 5), 'f1': round(f1_v, 5), 'acc': round(acc_v, 5)})

    roc_pts_sorted = sorted(roc_curve, key=lambda d: d['fpr'])
    roc_auc = 0.0
    for i in range(1, len(roc_pts_sorted)):
        dx = roc_pts_sorted[i]['fpr'] - roc_pts_sorted[i - 1]['fpr']
        roc_auc += 0.5 * (roc_pts_sorted[i]['tpr'] + roc_pts_sorted[i - 1]['tpr']) * dx

    best_op = max(pr_f1_curve, key=lambda d: d['f1'])
    op_050 = min(pr_f1_curve, key=lambda d: abs(d['threshold'] - 0.5))
    eval_summary = {
        'model': 'SegFormer-B2 (nvidia/segformer-b2-finetuned-cityscapes-1024-1024 fine-tuned on KITTI data_road)',
        'classes': ['Non-Road', 'Other Road', 'Ego-Lane'],
        'val_images': len(val_samples),
        'train_images': len(train_samples),
        'best_epoch': max(epoch_rows, key=lambda r: r['val_road_f1'])['epoch'],
        'road_vs_nonroad': {
            'roc_auc': round(roc_auc, 5),
            'at_threshold_0_50': op_050,
            'at_peak_f1': best_op,
            'road_iou': max(epoch_rows, key=lambda r: r['val_road_f1'])['val_road_iou'],
            'miou': max(epoch_rows, key=lambda r: r['val_road_f1'])['val_miou'],
        },
        'ego_lane_seg': {
            'ego_lane_f1': round(2 * tp_l / max(1, 2 * tp_l + fp_l + fn_l), 5),
            'ego_lane_iou': round(tp_l / max(1, tp_l + fp_l + fn_l), 5),
            'precision': round(tp_l / max(1, tp_l + fp_l), 5),
            'recall': round(tp_l / max(1, tp_l + fn_l), 5),
        },
        'roc_curve_points': roc_curve[::4],
        'f1_pr_threshold_points': pr_f1_curve,
    }

    # Inference on KITTI Tracking sequences 0015 (Val) & 0019 (Test)
    track_root = find_kitti_tracking_root()
    calib_zip = ensure_calib_zip()
    seq_stats = {}
    if track_root is not None and calib_zip is not None:
        with zipfile.ZipFile(str(calib_zip), 'r') as zf:
            calib_entries = {Path(n).stem: zf.read(n).decode('utf-8') for n in zf.namelist() if n.endswith('.txt')}

        for seq_id in ('0015', '0019'):
            img_dir = track_root / 'image_02' / seq_id
            imgs = sorted(img_dir.glob('*.png'))
            p2 = parse_p2_from_calib_bytes(calib_entries.get(seq_id, '')) or (721.5, 721.5, 609.5, 172.8)
            mapper = P2RoadMapper(*p2, camera_height_m=1.65)
            proc = TemporalRoadLaneProcessor(mapper=mapper, ema_alpha=0.45)

            frames_by_idx: Dict[str, Dict] = {}
            bs = 8
            road_ratios = []
            marking_fit_cnt = 0
            for start in range(0, len(imgs), bs):
                batch_paths = imgs[start : start + bs]
                batch_bgrs = [cv2.imread(str(p)) for p in batch_paths]
                fh, fw = batch_bgrs[0].shape[:2]
                tensors = []
                for bgr in batch_bgrs:
                    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H))
                    arr = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                    tensors.append(np.transpose(arr, (2, 0, 1)))
                x_t = torch.from_numpy(np.stack(tensors)).to(device)
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = F.interpolate(model(pixel_values=x_t).logits, size=(fh, fw), mode='bilinear', align_corners=False)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, 3, fh, fw]
                for p_img, bgr, pr in zip(batch_paths, batch_bgrs, probs):
                    fr_idx = int(p_img.stem)
                    road_prob = np.clip(pr[1] + pr[2], 0.0, 1.0)
                    ego_prob = np.clip(pr[2], 0.0, 1.0)
                    rec = proc.extract_frame_road_lane_record(fr_idx, bgr, road_prob, ego_prob, rle_downsample=8)
                    frames_by_idx[str(fr_idx)] = rec
                    road_ratios.append(rec['ground_road_ratio'])
                    if rec['lane_source'] in ('marking_fit', 'hybrid_fit'):
                        marking_fit_cnt += 1

                    # Save visual diagnostic image for frame 45 and 55
                    if fr_idx in (45, 55):
                        vis = bgr.copy()
                        sem = vis.copy()
                        for poly in rec['road_px_polys']:
                            cv2.fillPoly(sem, [np.array(poly, dtype=np.int32)], (55, 185, 70))
                        for poly in rec['ego_lane_px_polys']:
                            cv2.fillPoly(sem, [np.array(poly, dtype=np.int32)], (210, 175, 45))
                        vis = cv2.addWeighted(sem, 0.28, vis, 0.72, 0.0)
                        for side_k, col in (('left_ego', (0, 235, 255)), ('right_ego', (255, 210, 60))):
                            if side_k in rec['lane_lines_px']:
                                cv2.polylines(vis, [np.array(rec['lane_lines_px'][side_k], dtype=np.int32)], False, col, 3, cv2.LINE_AA)
                        cv2.imwrite(str(OUT_DIR / f'road_lane_preview_{seq_id}_f{fr_idx:04d}.png'), vis)

            rl_out = {
                'sequence_id': seq_id,
                'road_backend': 'segformer-b2-cityscapes-kitti-road',
                'frames_count': len(frames_by_idx),
                'frames_by_idx': frames_by_idx,
            }
            (OUT_DIR / f'road_lane_{seq_id}.json').write_text(json.dumps(rl_out), encoding='utf-8')
            seq_stats[seq_id] = {
                'frames': len(frames_by_idx),
                'mean_ground_road_ratio': round(float(np.mean(road_ratios)), 4),
                'marking_or_hybrid_lane_rate': round(marking_fit_cnt / max(1, len(frames_by_idx)), 4),
            }
            print(f'[INFER {seq_id}] frames={len(frames_by_idx)} json_saved', flush=True)

    eval_summary['tracking_inference'] = seq_stats
    (OUT_DIR / 'road_eval_metrics.json').write_text(json.dumps(eval_summary, indent=2), encoding='utf-8')
    print('[DONE] road_eval_metrics:', json.dumps(eval_summary['road_vs_nonroad']['at_threshold_0_50']), 'AUC=', eval_summary['road_vs_nonroad']['roc_auc'], flush=True)


if __name__ == '__main__':
    main()
