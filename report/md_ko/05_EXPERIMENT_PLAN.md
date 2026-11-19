# 실험 계획 및 정량 벤치마크: 카메라 중심 시계열 자차 중심 World Model

## 1. 연구 가설 (Research Hypotheses)

- **H1 (시퀀스 비독립 무결성 및 마스킹)**: 시퀀스 비독립 분할과 `DontCare` + Distractor(`Van`, `Person`, `Person_sitting`) 마스킹을 결합하면 비마스킹 또는 프레임 단위 무작위 분할 대비 시계열 과적합과 잘못된 False-Negative 경사를 방지할 수 있습니다.
- **H2 (시계열 연속성)**: 2D 검출 결과를 ByteTrack(`src/track_utils.py`)으로 연결하고 운동학 칼만 필터(`src/world_model.py`)로 자차 중심 궤적을 필터링하면 단안 지면 깊이 추정의 잡음을 안정화하고 짧은 가려짐(Occlusion) 구간에서도 객체 정체성을 유지할 수 있습니다.
- **H3 (불확실성 기반 예측 및 위험도 평가)**: 검출, 깊이, 추적, 운동 예측의 불확실성을 분리 모델링하면 단순 선형 BBox 외삽 대비 보정된 다중 시점 궤적 범위(`t + 0.5s` ~ `+2.0s`)와 일관된 위험도 지표를 산출할 수 있습니다.

---

## 2. 베이스라인 및 절제 연구(Ablation Studies)

### 베이스라인 모델
1. **B0 — 프레임 독립 검출 (`src/detect.py`)**: 시계열 ID나 자차 중심 공간 투영이 없는 프레임별 2D 바운딩 박스 검출.
2. **B1 — 검출기 + 2D 추적기 (`YOLOv8s + ByteTrack`, `src/track_utils.py`)**: `px/sec` 속도만 사용하는 2D 이미지 평면 추적.
3. **B2 — 시계열 자차 중심 World Model (`src/world_model.py`)**: 보정된 `P2` 지면 투영 + 4D 운동학 칼만 필터 + `Observed/Inferred/Occluded/Unknown` 상태 머신 + Safety Shield.

### 계획된 절제 실험
| 실험 항목 | 비교 변형 | 목표 관측 지표 |
|---|---|---|
| **A1: 라벨 마스킹 정책** | `raw` vs `dontcare_only` vs `dontcare_and_distractors` | `Car`/`Pedestrian` 정밀도 및 승합차·앉은 보행자 오검출률 변화 |
| **A2: 추적기 매칭 알고리즘** | `bytetrack.yaml` vs `botsort.yaml` | 추적 연속성(`IDF1`, `IDSW`) 및 Coasting 안정성 |
| **A3: 운동 모델** | 단순 선형 외삽 vs 4D 칼만 필터 (`accel_sigma` 튜닝) | `+0.5s .. +2.0s` 다중 시점 `ADE` / `FDE` |

---

## 3. 평가 지표 및 정답(GT) 가용성

| 모듈 | 평가지표 | KITTI Tracking 정답(GT) 가용 여부 | 평가 도구 |
|---|---|---|---|
| **검출 (Detection)** | Precision, Recall, `mAP@0.5`, `mAP@0.5:0.95` | **가용** (`label_02` 2D 박스) | `ultralytics` `val` (Val & Test 분할) |
| **추적 (Tracking)** | HOTA, IDF1, ID Switches | **가용** (`label_02` 2D + `track_id`, 공식 `Car`/`Pedestrian`) | TrackEval / `kitti_preds/<seq>.txt` |
| **단안 깊이 (Depth)** | `Abs Rel`, `RMSE` (m) | **가용** (`label_02` cols 13–15 3D `x, y, z`) | `evaluate_world_model_against_kitti_gt` |
| **궤적 예측 (Prediction)** | `ADE`, `FDE` (m, `0.5s–2.0s`) | **가용** (미래 시점 `label_02` 3D 위치) | `evaluate_world_model_against_kitti_gt` |
| **위험도 평가 (Risk)** | 위험도 지표 분포, 보정(Calibration) | KITTI Tracking 내 **직접 GT 없음** | 대리 재생(Proxy Replay) 감사(휴리스틱 명기) |
| **안전 개입 (Control)** | 충돌률, 개입률 | **불가** (오프라인 영상 전용, 폐루프 시뮬레이터 없음) | 오프라인 권고 로그만 기록 |

---

## 4. 실험 실행 조건

- **실행 환경**: Kaggle CLI GPU (`NvidiaTeslaT4`, `2x T4` 자동 DDP 지원) 및 Google Colab CLI 연동.
- **타겟 부류**: `Car`, `Pedestrian`, `Cyclist` (`nc = 3`).
- **v1 베이스라인 하이퍼파라미터**: `yolov8s.pt`, `epochs=30`, `imgsz=640`, `batch=16`, `patience=10`, `device=0`.
- **v2 개선 하이퍼파라미터**: `yolo11m.pt`, `epochs=40` (24 epoch 조기 종료, Best 12 epoch), `imgsz=960`, `batch=16` (`8 x 2 GPUs`), `patience=12`, `cos_lr=True`, `close_mosaic=10`, `mixup=0.1`, `oversample_rare_classes=True` (`Cyclist x4`, `Pedestrian x2`, `6,974` 가중 학습 엔트리; `val`/`test` 미변조).
- **추적·마스킹·World Model 파라미터**: `ByteTrack` @ `imgsz=960`, `conf=0.25`, `mask_model=mobile_sam.pt`, `fps=10.0`, `camera_height_m=1.65`, `max_coast_sec=1.0`, `horizons=(0.5, 1.0, 1.5, 2.0)`, `CollisionRiskSmoother` (`alpha_up=0.36`, `alpha_down=0.22`, `max_rise_per_sec=1.10`, `max_fall_per_sec=0.85`, `hysteresis_margin=0.04`).

---

## 5. 핵심 연구 질문(RQ1–RQ3) 초기 검증 결과 (`src/validation_experiments.py`, Kaggle v5)

`training/` 시퀀스 `0015` (376 frames) 및 `0019` (1059 frames)를 대상으로 [src/validation_experiments.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/validation_experiments.py)로 평가했습니다([validation_rq1_rq2_rq3.json](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/world_model/world_outputs/validation_rq1_rq2_rq3.json)).

### RQ1: 좌표 변환 및 자차 움직임 보상
| 지표 | 시퀀스 `0015` (376 frames) | 시퀀스 `0019` (1059 frames) |
|---|---|---|
| 자차 주행 거리 (`oxts`) | `71.61 m` | `238.10 m` |
| 최대 `ego ↔ world` 왕복 오차 | `2.161e-14 m` | `3.352e-14 m` |
| 정지 GT 트랙 / 매칭 샘플 수 | `5` 트랙 / `369` 샘플 | `36` 트랙 / `602` 샘플 |
| 정지 객체 겉보기 속도 (비보정 자차 상대 평균) | `1.409 m/s` | `3.370 m/s` |
| 정지 객체 겉보기 속도 (OXTS World 보상 평균) | `0.618 m/s` (`-56.1%`) | `2.350 m/s` (`-30.3%`) |

### RQ2: 미래 궤적 예측 (`+0.5s` ~ `+2.0s`) vs 실제 미래 GT 관측치
| 모델 변형 | 시퀀스 `0015` Future-Ego ADE / FDE (m) | 시퀀스 `0019` Future-Ego ADE / FDE (m) |
|---|---|---|
| **Zero-Velocity 베이스라인** | `10.1950 / 11.4531` | `4.0831 / 5.2281` |
| **자차 상대 칼만 (No OXTS)** | `8.7443 / 9.3002` | `3.6625 / 4.7413` |
| **OXTS World 칼만 + CTRV 자차** | **`8.3848 / 8.8193`** (World `8.3388 / 8.6873`) | **`3.4873 / 4.4877`** (World `3.3824 / 4.2587`) |

- **불확실성 보정 (2D 마할라노비스 `1σ / 2σ`, 이론치 `0.3935 / 0.8647`)**: `0019` `0.4138 / 0.6149` (`9,156` 샘플); `0015` `0.1451 / 0.3602` (`3,301` 샘플, 단안 지면 깊이 RMSE `12.484 m` vs `0019` `4.2726 m` 영향).

### RQ3: 인식·예측 불확실성이 위험도 및 Safety Shield에 미치는 인과성
| 반사실 실험군 (Arm) | 시퀀스 `0015` 평균 위험도 / 비정상 주행 비율 | 시퀀스 `0019` 평균 위험도 / 비정상 주행 비율 |
|---|---|---|
| **Arm A: 불확실성 미고려 (Blind)** | `0.3076 / 41.76%` (`Emergency=0`) | `0.5325 / 84.70%` (`Emergency=98`) |
| **Arm B: 불확실성 반영 (Nominal)** | `0.5189 / 90.43%` (`248/376` 상향, `Emergency=10`) | `0.6990 / 89.52%` (`377/1059` 상향, `Emergency=227`) |
| **Arm C: 불확실성 반영 (Degraded)** | `0.5428 / 90.96%` (`254/376` 위험 상승, `23` 상향) | `0.7003 / 89.14%` (`592/1059` 위험 상승, `77` 상향, `Emergency=242`) |

---

## 6. v2 업그레이드 검증 (`v1/v5 베이스라인` vs `v2 개선판`)

모든 v2 산출물은 Kaggle `2x T4 GPU`에서 실행되어 [results/v2_improved/outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v2_improved/outputs)에 저장되었습니다.

### 6.1 Stage 2 객체 검출 정확도 (`eval_metrics.json`)
| Split / 지표 | v1 베이스라인 (`yolov8s @ 640`) | v2 개선판 (`yolo11m @ 960 + Oversampling`) | 개선폭 (`Delta`) |
|---|---:|---:|---:|
| **Val `mAP@0.5` / `mAP@0.5:0.95`** | `0.4521 / 0.2471` | **`0.5235 / 0.3060`** | **`+0.0714 (+15.8%)` / `+0.0589`** |
| **Val AP50 (`Car / Ped / Cyc`)** | `0.7501 / 0.5141 / 0.0922` | **`0.8235 / 0.5871 / 0.1598`** | `+0.0734 / +0.0730 / +0.0676 (+73.3%)` |
| **Test `mAP@0.5` / `mAP@0.5:0.95`** | `0.3652 / 0.1695` | **`0.4269 / 0.2145`** | **`+0.0617 (+16.9%)` / `+0.0450`** |
| **Test AP50 (`Car / Ped / Cyc`)** | `0.6294 / 0.3497 / 0.1165` | **`0.6662 / 0.3631 / 0.2513`** | `+0.0368 / +0.0134 / +0.1348 (+115.7%)` |

### 6.2 Stage 2–3 SAM 인스턴스 객체 마스킹 (`tracking_summary.json`)
- **마스크 백엔드**: ByteTrack 트랙에 Box-Prompt를 적용한 `mobile_sam.pt` (`trajectories_masks.json` 및 CSV `mask_area_px`, `mask_fill_ratio`, `mask_bottom_x/y`).
- **시퀀스 `0015` (376 frames)**: `1,370`개 마스킹 인스턴스 (`41`개 고유 트랙).
- **시퀀스 `0019` (1059 frames)**: `3,788`개 마스킹 인스턴스 (`203`개 고유 트랙).

### 6.3 Stage 4–5 World Model 깊이, 자차 보상 및 미래 예측 (`validation_0015.json`, `validation_0019.json`)
| 지표 (전체 시퀀스) | 시퀀스 `0015` v5 → v2 | 시퀀스 `0019` v5 → v2 |
|---|---|---|
| **RQ1 정지체 World 속도 평균** | `0.618 → 0.264 m/s` (`-57.3%`) | `2.350 → 0.834 m/s` (`-64.5%`) |
| **RQ2 깊이 `Abs Rel / RMSE`** | `0.5207 / 12.484m → 0.2720 / 2.756m` (`-77.9%` RMSE) | `0.1803 / 4.2726m → 0.1560 / 3.3568m` (`-21.4%` RMSE) |
| **RQ2 Future-Ego `ADE / FDE`** | `8.3848 / 8.8193m → 3.4999 / 4.5694m` (`-58.3%` ADE) | `3.4873 / 4.4877m → 2.7907 / 3.2671m` (`-20.0%` ADE) |
| **RQ2 2D 불확실성 보정 (`1σ / 2σ`)** | `0.1451 / 0.3602 → 0.4584 / 0.6941` | `0.4138 / 0.6149 → 0.5808 / 0.7500` |
| **RQ3 Blind → Nominal → Degraded 평균 위험도** | `0.4637 → 0.6498 → 0.6898` (`258` / `334` frames) | `0.7518 → 0.9053 → 0.9107` (`609` / `595` frames) |

### 6.4 RQ4: 충돌 감지기 기울기 평활화 (`validation_0015.json`, `validation_0019.json`)
| 시퀀스 | 지표 | 비평활화 (Raw Potential) | 기울기 평활화 (`CollisionRiskSmoother`) | 감소율 |
|---|---|---:|---:|---:|
| **`0015` (376f)** | `mean |dR/dt|` / `p95 |dR/dt|` / `max |dR/dt|` | `0.1656 / 0.7430 / 2.2500 /s` | `0.1290 / 0.4130 / 1.0400 /s` | **`-22.10% / -44.41% / -53.78%`** |
| **`0015` (376f)** | 스파이크 (`> 1.5/s`) / 판단 플립 | `3` / `24` | **`0`** / `18` | **`-100.0%` / `-25.0%`** |
| **`0019` (1059f)** | `mean |dR/dt|` / `p95 |dR/dt|` / `max |dR/dt|` | `0.2236 / 0.9000 / 10.0000 /s` | `0.1188 / 0.5200 / 1.2500 /s` | **`-46.87% / -42.22% / -87.50%`** |
| **`0019` (1059f)** | 스파이크 (`> 1.5/s`) / 판단 플립 | `23` / `74` | **`0`** / `40` | **`-100.0%` / `-45.95%`** |

---

## 7. v3 오른쪽 BEV 시각화 및 오경보 수정 (`v2_improved` → `v3_bev_fixed`, Kaggle v7)

산출물은 [results/v3_bev_fixed/world_outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v3_bev_fixed/world_outputs)에 보관되어 있습니다(`mask_backend: "mobile_sam.pt"`).

### 7.1 오경보 억제 및 개입 정밀도 (`Arm B` Nominal)
| 시퀀스 | 지표 | v2 수정 전 | v3 수정 후 | 변화량 (`Delta`) |
|---|---|---:|---:|---:|
| **`0015` (376f)** | 평균 위험도 / 비정상 주행 비율 | `0.6498 / 95.21%` | **`0.2791 / 47.07%`** | **`-57.0%` 위험도 / `-48.14%p`** |
| **`0015` (376f)** | Normal Driving / Emergency 프레임 수 | `18 / 104` | **`199 / 16`** | **`+181 Normal` / `-84.6% Emergency`** |
| **`0019` (1059f)** | 평균 위험도 / Emergency 프레임 수 | `0.9053 / 849` | **`0.6290 / 216`** | **`-30.5%` 위험도 / `-74.6% Emergency`** |

### 7.2 유지·개선된 RQ1–RQ4 검증 지표 (`v3_bev_fixed`)
| RQ | 시퀀스 `0015` (376f) | 시퀀스 `0019` (1059f) |
|---|---|---|
| **RQ1 왕복 오차 / 정지체 World 속도** | `2.161e-14 m` / `0.227 m/s` (비보정 `1.358 m/s`) | `3.352e-14 m` / `0.756 m/s` (비보정 `1.849 m/s`) |
| **RQ2 OXTS+CTRV `ADE / FDE` (vs No-OXTS / Zero-Vel)** | `3.6052 / 4.6872 m` (vs `4.275 / 5.6099`, `5.2465 / 7.1198`) | `2.7890 / 3.2453 m` (vs `2.9277 / 3.5301`, `3.2491 / 4.2215`) |
| **RQ2 2D 실측 커버리지 (`1σ / 2σ`)** | `0.3939 / 0.5779` (이론치 `0.3935 / 0.8647`) | `0.4863 / 0.6826` |
| **RQ3 Blind → Nominal → Degraded 평균 위험도** | `0.2220 → 0.2791 → 0.2792` (`77 / 18` 상향, `259/376` 상승) | `0.5242 → 0.6290 → 0.6396` (`521 / 121` 상향, `732/1059` 상승) |
| **RQ4 최대 `|dR/dt|` / 스파이크(`>1.5/s`) / 플립** | `5.29 → 1.10 /s (-79.21%)` / `5 → 0 (-100%)` / `34 → 22 (-35.29%)` | `5.76 → 1.10 /s (-80.90%)` / `42 → 0 (-100%)` / `140 → 78 (-44.29%)` |
