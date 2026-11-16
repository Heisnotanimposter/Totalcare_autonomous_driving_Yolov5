# Architecture: Camera-Centric Temporal Ego-centric World Model

## 1. System Overview

This project implements a 7-stage pipeline transforming a single forward-facing camera stream into a structured **Temporal Ego-centric World Model**, multi-horizon trajectory predictions, risk indicators, and an advisory Safety Shield output:

```text
[Stage 1: Input Video (KITTI 10 Hz)]
        │
        ▼
[Stage 2: Visual Perception (src/kitti_prep.py & YOLO11m @ 960px + MobileSAM)]
        │  2D BBox, Class (Car, Pedestrian, Cyclist), Detection Confidence
        │  Rare-Class Train Oversampling (Cyclist x4, Pedestrian x2) + Instance Polygons
        ▼
[Stage 3: Temporal Understanding (src/track_utils.py, ByteTrack @ 960px + MobileSAM)]
        │  Sequence-scoped Track ID, Frame ID, Timestamp, mask_area_px, mask_fill_ratio, mask_bottom_xy
        ▼
[Stage 4: Ego-centric World Model (src/world_model.py)]
        │  P2 Inverse-Variance Depth Fusion (Flat Ground + Class-Height Prior + Mask Bottom Contact)
        │  OXTS IMU/GPS Ego-Motion Compensation (EgoPose2D: Ego <-> World ENU)
        │  State Machine: Past <-> Present <-> Future | Observed / Inferred / Occluded / Unknown
        ▼
[Stage 5: Future Prediction (src/world_model.py, World-ENU Kalman + CTRV Ego)]
        │  t + 0.5s, 1.0s, 1.5s, 2.0s Means + Confidence/Mask-Scaled 1-sigma Covariance Ellipses
        ▼
[Stage 6: Risk Assessment (src/world_model.py, C1 Potential + CollisionRiskSmoother)]
        │  Smooth Sigmoid Hazard Potential + Asymmetric EMA + Slew-Rate |dR/dt| Clamp + Calibrated TTC
        ▼
[Stage 7: Safety-aware Control (src/world_model.py & src/world_visualizer.py)]
           Hysteresis-Debounced Advisory Safety Shield + Split-Screen (Masks + BEV World)
```

---

## 2. Coordinate Systems & Transformations

### 2.1 KITTI Camera Coordinate Frame
In KITTI Tracking (`calib/XXXX.txt`, `label_02/XXXX.txt`):
- $X_{\text{cam}}$: right ($+$)
- $Y_{\text{cam}}$: down ($+$)
- $Z_{\text{cam}}$: forward along optical axis ($+$)

### 2.2 Ego-centric BEV Coordinate Frame
Defined in `KittiCalibration` ([src/world_model.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_model.py)):
- $X_{\text{ego}}$: lateral right ($+$), meters
- $Y_{\text{ego}}$: longitudinal forward ($+$), meters
- $Z_{\text{ego}}$: height upward ($+$), meters

Under nominal level mounting ($h_{\text{cam}} = 1.65\text{ m}$, zero pitch/roll assumption):
$$X_{\text{ego}} = X_{\text{cam}}, \quad Y_{\text{ego}} = Z_{\text{cam}}, \quad Z_{\text{ego}} = h_{\text{cam}} - Y_{\text{cam}}$$

### 2.3 Calibrated Inverse-Variance Depth Fusion & Analytical Uncertainty
Given KITTI intrinsics $P_2$ ($f_x, f_y, c_x, c_y$), contact point $(u_{\text{mid}}, v_{\text{bottom}})$ (using MobileSAM foreground bottom `mask_bottom_xy` when `mask_fill_ratio >= 0.25`, else the 2D box bottom-center), and class canonical height prior $H_c$ (`CLASS_HEIGHT_PRIORS_M`: `Car=1.52m`, `Pedestrian=1.72m`, `Cyclist=1.68m`):
1. **Flat-Ground Projection ($Y_{\text{ground}}$)** with pitch uncertainty $\sigma_\theta = 0.012\text{ rad}$:
   $$Y_{\text{ground}} = \frac{f_y \cdot h_{\text{cam}}}{v_{\text{bottom}} - c_y}, \qquad \sigma_{\text{ground}} = \sqrt{\left(\frac{Y_{\text{ground}}^2}{f_y h_{\text{cam}}}\sigma_v\right)^2 + \left(\frac{Y_{\text{ground}}^2}{h_{\text{cam}}}\sigma_\theta\right)^2}$$
2. **Pitch-Invariant Height-Prior Depth ($Y_{\text{height}}$)** on non-truncated boxes:
   $$Y_{\text{height}} = \frac{f_y \cdot H_c}{h_{\text{px}}}, \qquad X_{\text{ego}} = \frac{(u_{\text{mid}} - c_x) \cdot Y_{\text{fused}}}{f_x}$$
3. **Inverse-Variance Fusion**:
   $$w_g = \sigma_{\text{ground}}^{-2}, \quad w_h = \sigma_{\text{height}}^{-2}, \qquad Y_{\text{fused}} = \frac{w_g Y_{\text{ground}} + w_h Y_{\text{height}}}{w_g + w_h}, \qquad \sigma_Y = (w_g + w_h)^{-1/2}$$
Measurement covariance $R_w$ in World-ENU Kalman filtering is further scaled inversely by detection confidence and MobileSAM mask fill quality (`conf`, `mask_fill_ratio`).

**Integrity Rules:**
- If $P_2$ calibration is missing, `TemporalWorldModel` sets `calibrated = False` and **refuses** to compute metric speed (`m/s`) or Time-to-Collision (`TTC`).
- Projected 3D locations are always categorized as **Inferred** (monocular ground-plane/height-fused estimate), never **Observed**.

---

## 3. Ego-Motion Compensation & Temporal World Model (`Past → Present → Future`)

Implemented in `EgoPose2D`, `KittiOxtsOdometry`, `TemporalWorldModel`, and `ObjectWorldTrack` ([src/world_model.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_model.py)):

1. **OXTS Ego-Motion Compensation (`Ego ↔ World ENU`)**:
   - `KittiOxtsOdometry` converts KITTI `oxts/<seq>.txt` `(lat, lon, yaw, vf, vl, wz)` into local metric **East-North (ENU)** coordinates via Mercator projection anchored at frame 0.
   - `EgoPose2D.ego_to_world_xy` and `world_to_ego_xy` transform instantaneous Ego-centric detections (`+X` right, `+Y` forward) into a stationary World-ENU frame (observed round-trip error `< 4e-14 m`).
   - Object velocities and 4D Kalman states `[X_w, Y_w, vX_w, vY_w]` are filtered in **World-ENU** (subtracting Ego translation and yaw rotation), while Ego's planned corridor uses a Constant Turn Rate and Velocity (`predict_ctrv`) rollout from `vf` and `wz`.

2. **Temporal State Transition (`Future_prev → Present → Past`)**:
   - At timestamp $t_k$, each active track's World-frame 4D kinematic Kalman state and covariance $P$ are propagated forward by $\Delta t = t_k - t_{k-1}$ and projected back into Ego $t_k$'s BEV frame.
   - Tracks matched to current detections are updated with rotated measurement covariance $R_w = R_{\text{ego}\to w} \, \text{diag}(\sigma_X^2, \sigma_Y^2) \, R_{\text{ego}\to w}^T$ and assigned `observation_state = "Observed"`.
   - Tracks not observed at $t_k$ within `max_coast_sec` ($1.0\text{ s}$) transition to `observation_state = "Occluded"` with exponential confidence decay $\exp(-\Delta t_{\text{miss}} / 0.6)$.
   - Tracks missing beyond `max_coast_sec` archive into `past_world.expired_tracks` with their `last_seen_ts` and World-anchored position history.
   - Because `Past`, `Present`, and `Future` are indexed by timestamp relative to $t_k$ and maintained in World-ENU, their semantics remain invariant under Ego rotation or reversing.

3. **Spatial Information States (`Observed / Inferred / Occluded / Unknown`)**:
   - **Observed**: Direct 2D camera measurements and active tracked detections inside the front camera FOV.
   - **Inferred**: Ground-plane BEV positions, kinematic velocities, Kalman future trajectories (`+0.5s` to `+2.0s`), and Risk Indicators.
   - **Occluded**: Angular shadow sectors ray-cast behind foreground objects (`compute_occlusion_sectors`) and temporarily coasting tracks.
   - **Unknown**: Regions behind Ego ($Y_{\text{ego}} < 0$) and outside the front camera horizontal FOV (`2 * atan2(W / 2, fx)`).

---

## 4. Safety Shield Policy & Collision Detector Gradient Smoothing (Advisory Only)

`assess_risk_and_safety_shield` combines:
1. **$C^1$-Continuous Differentiable Risk Potential**: Per-object hazard scores use smooth `_sigmoid` curves over true **lateral covariance** (`lateral_uncertainty_m`, `cov_future_ego[0][0]`) corridor intrusion bounds, approach-gated longitudinal distance (`approach_w`, preventing stationary-vehicle and sidewalk false alarms), calibrated TTC, and compact earliest-horizon `conflict_zones` (`[1.0, 2.2] m`, max 1 per track), aggregated via probabilistic soft-union $1 - \prod_i (1 - r_i)$.
2. **Aspect-Corrected BEV Uncertainty Ellipses (`BevCanvasMapper.cov_to_pixel_ellipse`)**: Projects World-ENU $2\times 2$ covariance into BEV pixel coordinates via Jacobian $J = \text{diag}(s_x, -s_y)$ so elongated longitudinal covariance is never stretched laterally across lanes, rendered as soft translucent horizon fills (`t = +1.0s, +2.0s`) and a single outer `+2.0s` contour.
3. **Gradient Smoothing (`CollisionRiskSmoother`)**: Eliminates high-frequency detector/depth spikes across frames via asymmetric EMA (`alpha_up=0.36`, `alpha_down=0.22`), explicit slew-rate clamping (`max_rise_per_sec=1.10/s`, `max_fall_per_sec=0.85/s`), per-track risk EMA (`per_track_ema`), low-TTC debouncing, and hysteresis (`hysteresis_margin=0.04`) across Safety Shield thresholds.

Outputs are strictly **advisory**:
- `Normal Driving`
- `Caution`
- `Speed Reduction`
- `Braking Recommendation`
- `Emergency Intervention`

