# Experiment Plan: Camera-Centric Temporal Ego-centric World Model

## 1. Research Hypotheses

- **H1 (Sequence-Disjoint Integrity & Masking):** Sequence-disjoint splitting combined with `DontCare` + distractor (`Van`, `Person`, `Person_sitting`) masking prevents temporal overfitting and spurious false-negative gradients compared to unmasked or frame-random splits.
- **H2 (Temporal Continuity):** Associating 2D detections with ByteTrack (`src/track_utils.py`) and filtering ego-centric trajectories via a kinematic Kalman filter (`src/world_model.py`) stabilizes noisy monocular ground-plane depth estimates and preserves object identity across brief occlusions.
- **H3 (Uncertainty-Aware Prediction & Risk):** Separating detection, depth, tracking, and motion prediction uncertainties yields calibrated multi-horizon trajectory bounds (`t + 0.5s` to `+2.0s`) and more consistent risk indicators than naive linear bounding-box extrapolation.

---

## 2. Baselines & Ablation Studies

### Baselines
1. **B0 — Frame-Independent Detection (`src/detect.py`)**: Per-frame 2D bounding boxes with no temporal identity or ego-centric spatial projection.
2. **B1 — Detector + 2D Tracker (`YOLOv8s + ByteTrack`, `src/track_utils.py`)**: 2D image-plane tracking with `px/sec` velocity only.
3. **B2 — Temporal Ego-centric World Model (`src/world_model.py`)**: Calibrated `P2` ground-plane projection + 4D kinematic Kalman filter + `Observed/Inferred/Occluded/Unknown` state machine + Safety Shield.

### Planned Ablations
| Study | Variants | Target Observation |
|---|---|---|
| **A1: Label Masking Policy** | `raw` vs `dontcare_only` vs `dontcare_and_distractors` | Impact on `Car`/`Pedestrian` precision and false-positive rate on vans/sitting persons |
| **A2: Tracker Association** | `bytetrack.yaml` vs `botsort.yaml` | Track continuity (`IDF1`, `IDSW`) and coasting stability |
| **A3: Motion Model** | Naive linear extrapolation vs 4D Kalman filter (`accel_sigma` tuned) | Multi-horizon `ADE` / `FDE` at `+0.5s .. +2.0s` |

---

## 3. Evaluation Metrics & Ground-Truth Availability

| Module | Metrics | KITTI Tracking GT Availability | Tooling |
|---|---|---|---|
| **Detection** | Precision, Recall, `mAP@0.5`, `mAP@0.5:0.95` | **Available** (`label_02` 2D boxes) | `ultralytics` `val` on Val & Test splits |
| **Tracking** | HOTA, IDF1, ID Switches | **Available** (`label_02` 2D + `track_id`, official `Car`/`Pedestrian` scope) | TrackEval / `kitti_preds/<seq>.txt` |
| **Depth (Monocular)** | `Abs Rel`, `RMSE` (m) | **Available** (`label_02` cols 13–15 3D `x, y, z`) | `evaluate_world_model_against_kitti_gt` |
| **Motion Prediction** | `ADE`, `FDE` (m, `0.5s–2.0s`) | **Available** (future `label_02` 3D positions) | `evaluate_world_model_against_kitti_gt` |
| **Risk Assessment** | Risk Indicator distribution, Calibration | **No direct GT label** in KITTI Tracking | Proxy replay audit only (labeled heuristic) |
| **Safety Intervention** | Collision Rate, Intervention Rate | **Not available** (offline video only, no closed-loop sim) | Offline advisory logging only |

---

## 4. Experimental Execution Conditions

- **Execution Environment**: Kaggle CLI GPU (`NvidiaTeslaT4`, auto-DP `2x T4` supported) and/or Google Colab CLI (local CPU lacks PyTorch/CUDA).
- **Target Classes**: `Car`, `Pedestrian`, `Cyclist` (`nc = 3`).
- **Training Hyperparameters (v1 Baseline)**: `yolov8s.pt`, `epochs=30`, `imgsz=640`, `batch=16`, `patience=10`, `device=0`.
- **Training Hyperparameters (v2 Improved)**: `yolo11m.pt`, `epochs=40` (early-stopped at 24, best epoch 12), `imgsz=960`, `batch=16` (`8 x 2 GPUs`), `patience=12`, `cos_lr=True`, `close_mosaic=10`, `mixup=0.1`, `oversample_rare_classes=True` (`Cyclist x4`, `Pedestrian x2`, `6,974` weighted train entries; `val`/`test` untouched).
- **Tracking, Masking & World Model Params**: `ByteTrack` @ `imgsz=960`, `conf=0.25`, `mask_model=mobile_sam.pt`, `fps=10.0`, `camera_height_m=1.65`, `max_coast_sec=1.0`, `horizons=(0.5, 1.0, 1.5, 2.0)`, `CollisionRiskSmoother` (`alpha_up=0.36`, `alpha_down=0.22`, `max_rise_per_sec=1.10`, `max_fall_per_sec=0.85`, `hysteresis_margin=0.04`).

---

## 5. Core Research Validation Results (`src/validation_experiments.py`, Kaggle v5)

Evaluated using [src/validation_experiments.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/validation_experiments.py) on `training/` sequences `0015` (376 frames) and `0019` (1059 frames); raw outputs stored in [validation_rq1_rq2_rq3.json](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/world_model/world_outputs/validation_rq1_rq2_rq3.json).

### RQ1: Coordinate Transform & Ego-Motion Compensation
| Metric | Seq `0015` (376 frames) | Seq `0019` (1059 frames) |
|---|---|---|
| Ego Travel Distance (`oxts`) | `71.61 m` | `238.10 m` |
| Max `ego ↔ world` Round-Trip Error | `2.161e-14 m` | `3.352e-14 m` |
| Stationary GT Tracks / Matched Samples | `5` tracks / `369` samples | `36` tracks / `602` samples |
| Apparent Speed on Stationary Objects (Uncompensated Ego-Rel Mean) | `1.409 m/s` | `3.370 m/s` |
| Apparent Speed on Stationary Objects (OXTS World-Comp Mean) | `0.618 m/s` (`-56.1%`) | `2.350 m/s` (`-30.3%`) |

### RQ2: Future Prediction (`+0.5s` to `+2.0s`) vs Real Future GT Observations
| Model Variant | Seq `0015` Future-Ego ADE / FDE (m) | Seq `0019` Future-Ego ADE / FDE (m) |
|---|---|---|
| **Zero-Velocity Baseline** | `10.1950 / 11.4531` | `4.0831 / 5.2281` |
| **Ego-Relative Kalman (No OXTS)** | `8.7443 / 9.3002` | `3.6625 / 4.7413` |
| **OXTS World Kalman + CTRV Ego** | **`8.3848 / 8.8193`** (World `8.3388 / 8.6873`) | **`3.4873 / 4.4877`** (World `3.3824 / 4.2587`) |

- **Uncertainty Calibration (2D Mahalanobis `1σ / 2σ` vs Theoretical `0.3935 / 0.8647`)**: `0019` achieves `0.4138 / 0.6149` (`9,156` samples); `0015` achieves `0.1451 / 0.3602` (`3,301` samples, limited by monocular ground-plane depth RMSE `12.484 m` vs `4.2726 m` on `0019`).

### RQ3: Epistemic & Aleatoric Uncertainty Causality on Risk & Safety Shield
| Counterfactual Arm | Seq `0015` Mean Risk / Non-Normal Rate | Seq `0019` Mean Risk / Non-Normal Rate |
|---|---|---|
| **Arm A: Uncertainty-Blind** | `0.3076 / 41.76%` (`Emergency=0`) | `0.5325 / 84.70%` (`Emergency=98`) |
| **Arm B: Uncertainty-Aware (Nominal)** | `0.5189 / 90.43%` (`248/376` escalated, `Emergency=10`) | `0.6990 / 89.52%` (`377/1059` escalated, `Emergency=227`) |
| **Arm C: Uncertainty-Aware (Degraded)** | `0.5428 / 90.96%` (`254/376` higher risk, `23` escalated) | `0.7003 / 89.14%` (`592/1059` higher risk, `77` escalated, `Emergency=242`) |

---

## 6. v2 Upgrade Verification (`v1/v5 Baseline` vs `v2 Improved`)

All v2 artifacts were executed on Kaggle 2x T4 GPU and stored in [results/v2_improved/outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v2_improved/outputs).

### 6.1 Stage 2 Detection Accuracy (`eval_metrics.json`)
| Split / Metric | v1 Baseline (`yolov8s @ 640`) | v2 Improved (`yolo11m @ 960 + Oversampling`) | Delta |
|---|---:|---:|---:|
| **Val `mAP@0.5` / `mAP@0.5:0.95`** | `0.4521 / 0.2471` | **`0.5235 / 0.3060`** | **`+0.0714 (+15.8%)` / `+0.0589`** |
| **Val AP50 (`Car / Ped / Cyc`)** | `0.7501 / 0.5141 / 0.0922` | **`0.8235 / 0.5871 / 0.1598`** | `+0.0734 / +0.0730 / +0.0676 (+73.3%)` |
| **Test `mAP@0.5` / `mAP@0.5:0.95`** | `0.3652 / 0.1695` | **`0.4269 / 0.2145`** | **`+0.0617 (+16.9%)` / `+0.0450`** |
| **Test AP50 (`Car / Ped / Cyc`)** | `0.6294 / 0.3497 / 0.1165` | **`0.6662 / 0.3631 / 0.2513`** | `+0.0368 / +0.0134 / +0.1348 (+115.7%)` |

### 6.2 Stage 2–3 SAM Instance Object Masking (`tracking_summary.json`)
- **Mask Backend**: `mobile_sam.pt` box-prompted over ByteTrack tracks (`trajectories_masks.json` + CSV `mask_area_px`, `mask_fill_ratio`, `mask_bottom_x/y`).
- **Seq `0015` (376 frames)**: `1,370` masked instances (`41` unique tracks).
- **Seq `0019` (1059 frames)**: `3,788` masked instances (`203` unique tracks).

### 6.3 Stage 4–5 World Model Depth, Ego-Motion & Future Prediction (`validation_0015.json`, `validation_0019.json`)
| Metric (Full Sequence) | Seq `0015` v5 → v2 | Seq `0019` v5 → v2 |
|---|---|---|
| **RQ1 Stationary World Speed Mean** | `0.618 → 0.264 m/s` (`-57.3%`) | `2.350 → 0.834 m/s` (`-64.5%`) |
| **RQ2 Depth `Abs Rel / RMSE`** | `0.5207 / 12.484m → 0.2720 / 2.756m` (`-77.9%` RMSE) | `0.1803 / 4.2726m → 0.1560 / 3.3568m` (`-21.4%` RMSE) |
| **RQ2 Future-Ego `ADE / FDE`** | `8.3848 / 8.8193m → 3.4999 / 4.5694m` (`-58.3%` ADE) | `3.4873 / 4.4877m → 2.7907 / 3.2671m` (`-20.0%` ADE) |
| **RQ2 2D Uncertainty Calibration (`1σ / 2σ`)** | `0.1451 / 0.3602 → 0.4584 / 0.6941` | `0.4138 / 0.6149 → 0.5808 / 0.7500` |
| **RQ3 Blind → Nominal → Degraded Mean Risk** | `0.4637 → 0.6498 → 0.6898` (`258` / `334` frames) | `0.7518 → 0.9053 → 0.9107` (`609` / `595` frames) |

### 6.4 RQ4: Collision Detector Gradient Smoothing (`validation_0015.json`, `validation_0019.json`)
| Sequence | Metric | Unsmoothed (Raw Potential) | Gradient-Smoothed (`CollisionRiskSmoother`) | Reduction |
|---|---|---:|---:|---:|
| **`0015` (376f)** | `mean |dR/dt|` / `p95 |dR/dt|` / `max |dR/dt|` | `0.1656 / 0.7430 / 2.2500 /s` | `0.1290 / 0.4130 / 1.0400 /s` | **`-22.10% / -44.41% / -53.78%`** |
| **`0015` (376f)** | Spikes (`> 1.5/s`) / Decision Flips | `3` / `24` | **`0`** / `18` | **`-100.0%` / `-25.0%`** |
| **`0019` (1059f)** | `mean |dR/dt|` / `p95 |dR/dt|` / `max |dR/dt|` | `0.2236 / 0.9000 / 10.0000 /s` | `0.1188 / 0.5200 / 1.2500 /s` | **`-46.87% / -42.22% / -87.50%`** |
| **`0019` (1059f)** | Spikes (`> 1.5/s`) / Decision Flips | `23` / `74` | **`0`** / `40` | **`-100.0%` / `-45.95%`** |

---

## 7. v3 Right-Panel BEV Visual & False-Positive Collision Fix (`v2_improved` → `v3_bev_fixed`, Kaggle v7)

Outputs archived in [results/v3_bev_fixed/world_outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v3_bev_fixed/world_outputs) (`mask_backend: "mobile_sam.pt"`).

### 7.1 False-Alarm Suppression & Intervention Precision (`Arm B` Nominal)
| Sequence | Metric | v2 Before Fix | v3 After Fix | Delta |
|---|---|---:|---:|---:|
| **`0015` (376f)** | Mean Risk Indicator / Non-Normal Rate | `0.6498 / 95.21%` | **`0.2791 / 47.07%`** | **`-57.0%` Risk / `-48.14%p`** |
| **`0015` (376f)** | Normal Driving / Emergency Intervention Frames | `18 / 104` | **`199 / 16`** | **`+181 Normal` / `-84.6% Emergency`** |
| **`0019` (1059f)** | Mean Risk Indicator / Emergency Intervention Frames | `0.9053 / 849` | **`0.6290 / 216`** | **`-30.5%` Risk / `-74.6% Emergency`** |

### 7.2 Preserved RQ1–RQ4 Validation Metrics (`v3_bev_fixed`)
| RQ | Sequence `0015` (376f) | Sequence `0019` (1059f) |
|---|---|---|
| **RQ1 Round-Trip / Stationary World Speed** | `2.161e-14 m` / `0.227 m/s` (vs uncomp `1.358 m/s`) | `3.352e-14 m` / `0.756 m/s` (vs uncomp `1.849 m/s`) |
| **RQ2 OXTS+CTRV `ADE / FDE` (vs No-OXTS / Zero-Vel)** | `3.6052 / 4.6872 m` (vs `4.275 / 5.6099`, `5.2465 / 7.1198`) | `2.7890 / 3.2453 m` (vs `2.9277 / 3.5301`, `3.2491 / 4.2215`) |
| **RQ2 2D Empirical Coverage (`1σ / 2σ`)** | `0.3939 / 0.5779` (theory `0.3935 / 0.8647`) | `0.4863 / 0.6826` |
| **RQ3 Blind → Nominal → Degraded Mean Risk** | `0.2220 → 0.2791 → 0.2792` (`77 / 18` escalated, `259/376` inc) | `0.5242 → 0.6290 → 0.6396` (`521 / 121` escalated, `732/1059` inc) |
| **RQ4 Max `|dR/dt|` / Spikes (`>1.5/s`) / Flips** | `5.29 → 1.10 /s (-79.21%)` / `5 → 0 (-100%)` / `34 → 22 (-35.29%)` | `5.76 → 1.10 /s (-80.90%)` / `42 → 0 (-100%)` / `140 → 78 (-44.29%)` |



