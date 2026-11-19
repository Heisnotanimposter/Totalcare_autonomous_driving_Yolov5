# Source, Kernel, Report, and Metrics Consolidated Index (Appendix)

This appendix provides a complete index of **source modules (`src/`)**, **remote GPU/CPU kernels (`kaggle/`, `kaggle_world_model/`, `kaggle_v4_*`, `colab/`)**, **reports & technical specs**, and **quantitative artifacts (`results/`, `logs/`)** across the repository.

---

## 1. Source Code Modules (`src/`)

| File | Lines | Core Role & Implementation |
|---|---:|---|
| `src/kitti_prep.py` | 624 | KITTI Zip/directory CRC & image decode integrity audit, sequence-disjoint 14/4/3 split, `DontCare`+Distractor (`Van`, `Person`, `Person_sitting`) 114-pixel masking, train-only rare-class oversampling (`train_weighted.txt`, `6,974` entries) |
| `src/track_utils.py` | 620 | ByteTrack (`imgsz=960`/`1280`, `conf=0.25`) multi-object tracking, `MobileSAM` (`mobile_sam.pt`) Box-Prompt instance segmentation (`trajectories_masks.json`), ground-contact point (`mask_bottom_xy`), KITTI 18-column TrackEval export |
| `src/road_lane.py` | — | `P2RoadMapper`, `parse_kitti_road_gt_mask`, `TemporalRoadLaneProcessor` (BEV IPM warp + HLS/Sobel lane detection + quadratic polynomial EMA smoothing), `query_object_road_lane_context` |
| `src/world_model.py` | 1,235+ | `P2` inverse-variance depth fusion (`ground_plane_bev_fusion`), `EgoPose2D` `SE(2)` OXTS ENU compensation & CTRV rollout, 4D World-ENU Kalman filter (`+0.5s~+2.0s`), `Observed/Inferred/Occluded/Unknown` state machine, lateral-covariance & road-context (`0.65` non-road attenuation) risk + `CollisionRiskSmoother` |
| `src/world_visualizer.py` | 710+ | `1360×632` split-screen renderer (`Left: Camera + MobileSAM + Road/Lane Masks` \| `Right: BEV Inferred World`), Jacobian $J=\text{diag}(s_x,-s_y)$ `cov_to_pixel_ellipse`, `--staged-demo` and `--full-overlay` MP4/keyframe export |
| `src/validation_experiments.py` | 499 | Automated **RQ1** (round-trip & stationary ego-motion compensation), **RQ2** (future prediction `OXTS+CTRV` vs `No-OXTS` vs `Zero-Vel` & 2D Mahalanobis coverage), **RQ3** (uncertainty causality `Arm A/B/C`), **RQ4** (gradient smoothing spike/flip reduction) |
| `src/dataset_prep.py`, `src/detect.py`, `src/train.py`, `src/export.py` | 377 | Baseline 9-class dashcam YOLO conversion, inference/risk banner visualization, YOLOv5/Ultralytics training & ONNX/TensorRT export utilities |

---

## 2. Remote GPU/CPU Training & Execution Kernels

| Kernel / Notebook | Environment | Executed Pipeline & Outputs |
|---|---|---|
| `kaggle/train_track_kitti.py` | Kaggle `2× Tesla T4` (15GB×2) | `test013/totalcare-kitti-yolo-train-and-track` (v2 `COMPLETE`): Audit → YOLO Export → `YOLO11m @ 960` DDP (24ep, ~108 min) → Val/Test Eval → `0015`/`0019` ByteTrack + MobileSAM → World Model validation |
| `kaggle_world_model/run_world_model_kaggle.py` | Kaggle CPU (`kernel_sources`) | `test013/totalcare-kitti-world-model-demo` (v7 `COMPLETE`): Mounts v2 `trajectories.csv` + `trajectories_masks.json` → `v3` BEV Jacobian ellipse & false-alarm fix re-render (4 MP4s) → RQ1–RQ4 verification |
| `kaggle_v4_road/train_road_lane_kitti.py` | Kaggle GPU (`Tesla P100`/`T4`) | `test013/totalcare-kitti-v4-road-lane` (`COMPLETE`): Fine-tunes `SegFormer-B2` 3-class head on `data_road` (`230/59`, 25ep) → infers `road_lane_0015.json` & `0019.json` |
| `kaggle_v4_detect/train_v4_detect_kitti.py` | Kaggle `2× Tesla T4` | `test013/totalcare-kitti-v4-detect` (`COMPLETE`): Deduped `kitti-stereo-image` merge (`14,371` train) → `YOLO11l @ 1280` (24ep) → Val/Test std & TTA + KITTI Easy/Mod/Hard eval → ByteTrack + MobileSAM |
| `kaggle_v4_world_model/run_v4_world_model_kaggle.py` | Kaggle CPU (`kernel_sources`) | `test013/totalcare-kitti-v4-world-model` (v2 `COMPLETE`): Integrates `v4` road/lane + `YOLO11l` tracks → renders 4 `v4` MP4s + RQ1–RQ4 validation |
| `colab/01_kitti_colab_audit_and_prep.ipynb`, `02_traffic_detect_track_colab.ipynb` | Google Colab CLI / Drive | `/content/drive/MyDrive/KITTI_Tracking/` Stage 1 audit, YOLO export, and tracking notebooks |

---

## 3. Reports & Technical Documentation

| Document | Summary |
|---|---|
| `FINAL_REPORT.md` | Final report mapped 1:1 to all 7 instructor evaluation items plus v4 road/lane & `0.9` accuracy results |
| `ML_LAYER_SPEC.md` | Layer-by-layer specification of `YOLO11m` (`Layer 0–23`), `MobileSAM`, `ByteTrack`, `SegFormer-B2`, `World Model & Safety Shield`, and 8 academic references |
| `architecture.md` | 7-stage system architecture, `P2` camera↔Ego equations, `Past→Present→Future` state transitions, and `CollisionRiskSmoother` design |
| `dataset_audit.md` | KITTI `8,008`-frame integrity audit, `DontCare`/Distractor masking, and 14/4/3 sequence-disjoint split |
| `experiment_plan.md` | Research hypotheses (H1–H3), baselines (B0–B2), `v1→v2→v3` benchmarks, and RQ1–RQ4 empirical results |
| `visualization_spec.md` | `1360×632` split-screen encoding spec and 5-second staged demonstration timeline |

---

## 4. Key Quantitative Metrics & Artifact Archive (`results/`, `logs/`)

| Category | Artifact Path | Key Quantitative Summary |
|---|---|---|
| **Stage 2 Detector (`v2`)** | `results/v2_improved/outputs/weights/best.pt` (`38.62 MB`) | `YOLO11m @ 960` (`231 layers`, `20.06M params`), **Val `mAP@0.5 = 0.5235`** (`+15.8%` vs `v1 0.4521`), **Test `mAP@0.5 = 0.4269`** |
| **Stage 4–7 Demo (`v3`)** | `results/v3_bev_fixed/world_outputs/world_model_*` | `0015` (`37.6s`, `11.58 MB`) / `0019` (`105.9s`, `53.48 MB`) + `8.0s` staged MP4s; `v3` Arm B risk `-57.0%` (`0015`) / `-30.5%` (`0019`), RQ4 spikes `-100%` |
| **v4 Road & Lane (`SegFormer-B2`)** | `results/v4_road_lane/road_outputs/road_lane_outputs/road_eval_metrics.json` | Road vs Non-Road **ROC AUC `0.9992`**, **Pixel Acc `0.9910`**, **Road F1 `0.9747`**, **Ego-Lane F1 `0.9119`** |
| **v4 Detector (`YOLO11l @ 1280`)** | `results/v4_road_lane/detect_outputs/outputs/eval_metrics_v4.json`, `weights/best.pt` (`49 MB`) | Val TTA **`mAP@0.5 = 0.7400`** (std `Precision 0.8685`), Test TTA `0.6705`, **Test Easy `mAP@0.5 = 0.8966`** (`~0.90`), **Test Mod `0.8404`** |
| **v4 World Model & RQ1–RQ4** | `results/v4_road_lane/world_outputs/v4_world_outputs/world_model_*` | `0015` (`37M`/`6.1M`) & `0019` (`25M`/`3.4M`) integrated road+lane+BEV demos; RQ4 spikes `8→0`, `16→0` (`-100%`), flips `-57.26%` |
| **Execution Logs** | `logs/v1_detect_track/`, `logs/v2_improved/`, `logs/v3_bev_fixed/` | Full Kaggle DDP training (`231 layers` summary) and World Model console logs |
