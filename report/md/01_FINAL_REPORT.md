# 프로젝트 최종 결과 보고서: Camera-Centric Temporal Ego-centric World Model

**GitHub 저장소:** https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5  
**데이터셋:** KITTI Object Tracking (`training/image_02`, `training/label_02`, `calib`, `oxts`)  
**실행 환경:** Kaggle CLI GPU (`2× Nvidia Tesla T4`, `Ultralytics 8.4.212`, `PyTorch 2.11.0+cu130`) + Google Colab CLI 연동

---

## 0. 기본 평가표 7개 항목 자체 점검 및 문서 매핑

| 평가 항목 | 충족 여부 | 근거 및 저장소 내 위치 |
|---|:---:|---|
| **1. 데이터 수집 (3개 부류 이상, 100개 이상 이미지)** | **충족** | `KITTI Tracking` 21개 시퀀스(`0000`–`0020`), 총 **`8,008`장** RGB 이미지, **`3`개 부류** (`Car`, `Pedestrian`, `Cyclist`). [`dataset_audit.md`](./dataset_audit.md) |
| **2. 객체 감지 및 추적 결과 5초 이상 동영상 제작** | **충족** | 좌우 분할 World Model 시연 영상 (`0015` staged `8.0초` / full **`37.6초`**, `0019` staged `8.0초` / full **`105.9초`**) + 2D ByteTrack·SAM 추적 영상. [`world_model_demo_0015.mp4`](./results/v3_bev_fixed/world_outputs/world_model_0015_full/world_model_demo_0015.mp4) |
| **3. 깃허브에 학습 모델과 시연 영상 저장 및 공개** | **커밋·푸시 대기** | 학습 모델 [`best.pt`](./results/v2_improved/outputs/weights/best.pt) (`38.62 MB`) 및 최종 `v3` 시연 영상 4종이 GitHub 100MB 제한 이내로 정리됨 (`.gitignore` 정비 완료) |
| **4. 내용을 결과 보고서에 포함** | **충족** | 본 보고서(1~5절)에 데이터 정제, 검출/세그멘테이션/추적, World Model & 안전 판단 전체 포함 |
| **5. 결과 보고서에 학습 과정 및 성능 벤치마크 포함** | **충족** | 제2–3절에 `v1 → v2 → v3` 학습 하이퍼파라미터, Epoch별 수렴 지표(`results.csv`), Detection/Tracking/Depth/Prediction/RQ1–RQ4 벤치마크 수록 |
| **6. 깃허브 링크 및 설명 포함** | **충족** | 상단 및 제4절에 저장소 URL, 디렉터리 구조, 재현 명령, 산출물 직접 경로 명시 |
| **7. 머신러닝 Layer별 기술설명서 및 참고자료 첨부** | **충족** | `YOLO11m` 전체 24개 모듈(`231 layers`), `MobileSAM`, `ByteTrack`, `World Model` Layer 상세 및 학술 참고자료 별첨 → [`ML_LAYER_SPEC.md`](./ML_LAYER_SPEC.md) |

---

## 1. 데이터 수집 및 시퀀스 비독립 전처리 (`Stage 1`)

- **원천 데이터 규모**: `KITTI Tracking` 21개 시퀀스(`0000`~`0020`), 총 **`8,008`장** (`1242×375` 등 4종 해상도, `10 Hz`, 무결성 검증 `ZipFile.testzip()` / `PIL.Image.verify()` 결손·손상 `0건`, [`kitti_step2_audit.json`](./results/v2_improved/outputs/kitti_step2_audit.json)).
- **타겟 부류(3개 클래스) 및 시퀀스 비독립 분할(Sequence-Disjoint Split)**: 동일 주행 시퀀스의 프레임이 Train/Val/Test에 섞이지 않도록 시퀀스 단위로 엄격 분리([`kitti_export_manifest.json`](./results/v2_improved/outputs/kitti_export_manifest.json)).

| Split | 시퀀스 수 | 프레임 수 | `Car` 박스 수 | `Pedestrian` 박스 수 | `Cyclist` 박스 수 | 비고 |
|---|---:|---:|---:|---:|---:|---|
| **Train** | 14 | `5,568` | `23,404` | `648` | `367` | 희소 클래스 오버샘플링(`train_weighted.txt`: `6,974` entries) |
| **Val** | 4 | `832` | `2,075` | `1,777` | `754` | `0002, 0012, 0015, 0017` |
| **Test** | 3 | `1,608` | `1,818` | `9,044` | `817` | `0013, 0016, 0019` (완전 미학습 주행) |
| **합계** | **21** | **`8,008`** | **`27,297`** | **`11,469`** | **`1,938`** | 총 `40,704` 타겟 객체 + `DontCare`/`Distractor` 마스킹 |

- **오류 경사 방지 마스킹**: 학습·평가 간 정합성을 위해 `DontCare` 및 인접 혼동 부류(`Van`, `Person`, `Person_sitting`) 영역을 회색 픽셀(`114`)로 마스킹하여 잘못된 False-Negative 페널티를 차단했습니다.

---

## 2. 학습 과정 및 개선 이력 (`v1 Baseline → v2 Improved → v3 BEV Fixed`)

### 2.1 모델 학습 설정 및 아키텍처 진화
1. **`v1` Baseline (`YOLOv8s @ 640px`)**: `epochs=30`, `batch=16`, `129 layers / 11.14M params / 28.6 GFLOPs`.
2. **`v2` Improved (`YOLO11m @ 960px` + Rare-Class Oversampling + `MobileSAM`)**:
   - 해상도를 `640 → 960`으로 상향하고 `C3k2 + C2PSA` 공간 어텐션 기반 `YOLO11m` (`231 layers / 20.06M params / 68.3 GFLOPs`) 채택.
   - Train split에 한정해 `Cyclist` / `Pedestrian` 포함 프레임을 가중 샘플링(`6,974` entries) 적용, `cos_lr=True`, `mixup=0.1`, `close_mosaic=10`, `patience=12` 설정으로 **`2× Tesla T4`**에서 DDP 학습 수행 (`24 epoch` 조기 종료, Best Checkpoint = **`Epoch 12`**, 약 108분 소요).
   - 추론 후단에 `mobile_sam.pt` (`38.8 MB`) Box-Prompt 세그멘테이션을 결합하여 객체 마스크 및 지면 접지점(`mask_bottom_xy`) 추출.
3. **`v3` BEV 시각·오경보 개선 (`world_model.py`, `world_visualizer.py`)**:
   - 오른쪽 BEV 패널에서 종방향 깊이 불확실성이 횡방향 차로와 타원을 2배 이상 왜곡·확대시키던 문제 수정(픽셀 야코비안 투영 $J=\text{diag}(s_x,-s_y)$, 횡방향 공분산 분리, 접근 속도 게이팅 `approach_w`, 트랙별 EMA 스무딩).

### 2.2 `v2 YOLO11m @ 960` Epoch별 학습 수렴 추이 ([`results.csv`](./results/v2_improved/outputs/train_results/results.csv))

| Epoch | 누적 시간 (s) | `train/box_loss` | `train/cls_loss` | `train/dfl_loss` | Val `Precision` | Val `Recall` | Val `mAP@0.5` | Val `mAP@0.5:0.95` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 214.4 | 1.1077 | 0.8017 | 1.0848 | 0.4067 | 0.4114 | 0.4074 | 0.2240 |
| **5** | 1097.3 | 0.9687 | 0.6330 | 1.0241 | 0.5063 | 0.4627 | 0.4516 | 0.2611 |
| **10** | 2197.2 | 0.8181 | 0.5107 | 0.9597 | 0.6701 | 0.4487 | 0.4886 | 0.2894 |
| **12 (Best)** | 2637.2 | 0.7793 | 0.4813 | 0.9431 | **0.7166** | **0.4814** | **0.5231** | **0.3059** |
| **19** | 4179.2 | 0.6791 | 0.4135 | 0.9019 | 0.7905 | 0.4846 | 0.5297 | 0.2985 |
| **24 (Stop)** | 5278.9 | 0.6194 | 0.3725 | 0.8823 | 0.6714 | 0.4341 | 0.4539 | 0.2738 |

- 학습 곡선 및 혼동 행렬 플롯: [`results.png`](./results/v2_improved/outputs/train_results/results.png), [`confusion_matrix_normalized.png`](./results/v2_improved/outputs/train_results/confusion_matrix_normalized.png)

---

## 3. 정량적 성능 벤치마크

### 3.1 Stage 2 객체 검출 성능 (`v1` vs `v2`, [`eval_metrics.json`](./results/v2_improved/outputs/eval_metrics.json))

| Split / Metric | `v1` (`YOLOv8s @ 640`) | `v2` (`YOLO11m @ 960`) | 개선폭 (`Δ`) |
|---|---:|---:|---:|
| **Val `mAP@0.5` / `mAP@0.5:0.95`** | `0.4521 / 0.2471` | **`0.5235 / 0.3060`** | **`+0.0714 (+15.8%)`** / `+0.0589` |
| **Val `Precision / Recall`** | `0.6026 / 0.4230` | **`0.7170 / 0.4822`** | `+0.1144` / `+0.0592` |
| **Val AP50 (`Car / Ped / Cyc`)** | `0.7501 / 0.5141 / 0.0922` | **`0.8235 / 0.5871 / 0.1598`** | `+0.0734 / +0.0730 / +0.0676 (+73.3%)` |
| **Test `mAP@0.5` / `mAP@0.5:0.95`** | `0.3652 / 0.1695` | **`0.4269 / 0.2145`** | **`+0.0617 (+16.9%)`** / `+0.0450` |
| **Test AP50 (`Car / Ped / Cyc`)** | `0.6294 / 0.3497 / 0.1165` | **`0.6662 / 0.3631 / 0.2513`** | `+0.0368 / +0.0134 /` **`+0.1348 (+115.7%)`** |

### 3.2 Stage 3 추적·인스턴스 마스킹 및 Stage 4–7 World Model (RQ1–RQ4) ([`validation_0015.json`](./results/v3_bev_fixed/world_outputs/validation_0015.json), [`validation_0019.json`](./results/v3_bev_fixed/world_outputs/validation_0019.json))

- **MobileSAM 마스킹**: 시퀀스 `0015` (376f) **`1,370`개** 인스턴스(`41`개 트랙), 시퀀스 `0019` (1059f) **`3,788`개** 인스턴스(`203`개 트랙) 추출 ([`tracking_summary.json`](./results/v2_improved/outputs/tracking_summary.json)).
- **연구 질문(RQ1–RQ4) 검증 요약 (`v3_bev_fixed`)**:

| 검증 항목 | 시퀀스 `0015` (376 frames, `37.6초`) | 시퀀스 `0019` (1059 frames, `105.9초`) |
|---|---|---|
| **RQ1 좌표·자차 보정 (`Ego↔World` 오차 / 정지체 World 속도)** | `2.161e-14 m` / **`0.227 m/s`** (비보정 `1.358 m/s`) | `3.352e-14 m` / **`0.756 m/s`** (비보정 `1.849 m/s`) |
| **RQ2 미래 예측 (`OXTS+CTRV` ADE/FDE vs Zero-Vel)** | **`3.6052 / 4.6872 m`** (vs Zero-Vel `5.2465 / 7.1198 m`) | **`2.7890 / 3.2453 m`** (vs Zero-Vel `3.2491 / 4.2215 m`) |
| **RQ2 2D 공분산 `1σ` 커버리지 (이론 `0.3935`)** | **`0.3939`** (이론치와 `0.0004` 차이) | **`0.4863`** |
| **RQ3 불확실성 인과성 (`Arm A Blind → B Nominal → C Degraded`)** | **`0.2220 → 0.2791 → 0.2792`** (`77 / 18`프레임 단계 상승) | **`0.5242 → 0.6290 → 0.6396`** (`521 / 121`프레임 단계 상승) |
| **RQ4 Gradient Smoothing (`max \|dR/dt\|` / 스파이크 / 플립)** | `5.29 → 1.10 /s (-79.2%)`, `>1.5/s` **`5 → 0 (-100%)`**, 플립 `34 → 22` | `5.76 → 1.10 /s (-80.9%)`, `>1.5/s` **`42 → 0 (-100%)`**, 플립 `140 → 78` |
| **`v2 → v3` 안전상황 오경보 억제 (`Arm B` 평균 Risk / Emergency)** | `0.6498 → 0.2791` (**`-57.0%`**), Emergency `104 → 16` (**`-84.6%`**) | `0.9053 → 0.6290` (**`-30.5%`**), Emergency `849 → 216` (**`-74.6%`**) |

### 3.3 시연 영상 및 주요 프레임 확인
- `Left Panel`: 원본 카메라 영상 + 검출 2D Box + `Car`/`Pedestrian`/`Cyclist` 라벨 + Track ID + Confidence + **MobileSAM 인스턴스 폴리곤 마스크**.
- `Right Panel`: Ego-centric BEV (`X` 좌우 / `Y` 전후), 자차 계획 궤적(CTRV), 과거 궤적(감쇠), 미래 예측 점선(`+0.5s~+2.0s`), 야코비안 보정 `1-sigma` 불확실성 타원, 폐색(Occluded) 부채꼴 및 후방 Unknown 음영, 소형 `CONFLICT` 링, 스무딩된 Safety Shield 권고 배너.
- 대표 검증 키프레임: [`seq_0015_frame_000045_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png), [`seq_0015_frame_000055_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000055_stage5.png), [`seq_0019_frame_000055_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png)

---

## 4. GitHub 저장소 안내 및 주요 산출물 링크

- **공식 GitHub 저장소:** https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5
- **핵심 소스 코드**: [`src/kitti_prep.py`](./src/kitti_prep.py), [`src/track_utils.py`](./src/track_utils.py), [`src/world_model.py`](./src/world_model.py), [`src/world_visualizer.py`](./src/world_visualizer.py), [`src/validation_experiments.py`](./src/validation_experiments.py)
- **학습된 최종 가중치**: [`best.pt`](./results/v2_improved/outputs/weights/best.pt) (`YOLO11m @ 960`, `38.62 MB`)
- **시연 동영상 (5초 이상)**:
  - Val 시퀀스 `0015` (8초 단계별 / 37.6초 전체): [`staged MP4`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/world_model_demo_0015.mp4) / [`full MP4`](./results/v3_bev_fixed/world_outputs/world_model_0015_full/world_model_demo_0015.mp4)
  - Test 시퀀스 `0019` (8초 단계별 / 105.9초 전체): [`staged MP4`](./results/v3_bev_fixed/world_outputs/world_model_0019_staged/world_model_demo_0019.mp4) / [`full MP4`](./results/v3_bev_fixed/world_outputs/world_model_0019_full/world_model_demo_0019.mp4)
- **원격 GPU 재현 커널**: [`kaggle/train_track_kitti.py`](./kaggle/train_track_kitti.py), [`kaggle_world_model/run_world_model_kaggle.py`](./kaggle_world_model/run_world_model_kaggle.py)

---

## 5. ML Layer별 기술설명서 및 참고자료 (요약)

전체 딥러닝·칼만·제어 레이어 명세 및 학술 문헌 목록은 별첨 기술설명서 [`ML_LAYER_SPEC.md`](./ML_LAYER_SPEC.md) 에 수록되어 있습니다.

1. **`YOLO11m` (`231 layers`, `20.06M params`, `68.3 GFLOPs`)**: Backbone (`Conv` Stem → `C3k2` CSP Bottleneck → `SPPF` → `C2PSA` Spatial Attention), Neck (`PAN-FPN`), Head (`Decoupled Anchor-Free Detect` + `DFL` + `CIoU` + `BCE`).
2. **`MobileSAM` (`38.8 MB`)**: Knowledge-Distilled `TinyViT` Image Encoder (`MBConv` + `Windowed Self-Attention`), Box Prompt Positional Encoder, Two-Way Cross-Attention Mask Decoder + `4×` Transposed Conv.
3. **`ByteTrack`**: 8D Constant-Velocity BBox Kalman Filter + 2단계 Hungarian Bipartite Matching (`High-Conf` + `Low-Conf` 가려짐 복구).
4. **Temporal World Model & Safety Shield**: `P2` 역분산 깊이 융합 + OXTS `SE(2)` 보상 + 4D World-ENU 칼만 필터 + 야코비안 BEV 공분산 타원 투영 + $C^1$ 위험 포텐셜 & `CollisionRiskSmoother` 슬루율/히스테리시스 평활화.
5. **`SegFormer-B2` (`nvidia/segformer-b2-finetuned-cityscapes-1024-1024`, v4 추가)**: MiT-B2 계층형 Mix-Transformer 인코더 + All-MLP 디코더 (`Non-Road` / `Other Road` / `Ego-Lane` 3-Class 헤드) + BEV IPM 역투영 및 HLS+Sobel 차선 2차 다항식 EMA 추적기.

---

## 6. Version 4 고도화: 도로/비도로 분류·차선 검출 및 정확도 `0.9` 달성 (`v4_road_lane`)

`v3_bev_fixed` 이후 사용자 추가 요구사항(**도로 vs 비도로 레이블 및 차선 검출·마스킹**, **정확도 `0.7 → 0.9` 개선**)을 반영하여 3개의 Kaggle 커널(`test013/totalcare-kitti-v4-road-lane`, `test013/totalcare-kitti-v4-detect`, `test013/totalcare-kitti-v4-world-model`)을 완주하고 [`results/v4_road_lane/`](./results/v4_road_lane/) 에 통합했습니다.

### 6.1 도로 vs 비도로 분류 및 자차로·차선 검출 (`SegFormer-B2` + BEV IPM)
- **데이터셋 및 학습**: `tryingit/roadlane-detection-evaluation-2013` (KITTI `data_road`, `230 train / 59 val` 계층 분할, 25 epochs)으로 `SegFormer-B2` 3-Class(`Non-Road`, `Other Road`, `Ego-Lane`) 미세조정.
- **실측 성능 ([`road_eval_metrics.json`](./results/v4_road_lane/road_outputs/road_lane_outputs/road_eval_metrics.json))**:
  - **Road vs Non-Road**: **ROC AUC = `0.9992`**, **Pixel Acc = `0.9910` (`99.10%`)**, **Road F1 = `0.9747`** (`Precision 0.9690` / `Recall 0.9804`), Road IoU `0.9506`, mIoU `0.9698`.
  - **Ego-Lane (`um_lane`)**: **F1 = `0.9119`** (`Precision 0.9070` / `Recall 0.9169`), IoU `0.8381`.
- **시각화 및 World Model 결합**: Left Panel에 도로(녹색)·비도로(자주색)·자차로(청록색) 반투명 마스킹, 좌/우 차선 2차 다항식 폴리라인, 객체별 `[LANE]`/`[RD]`/`[NON-RD]` 컨텍스트 배지를 표시하고, Right Panel BEV에 도로 폴리곤·곡률 차선과 비도로(`0.65` 위험 감쇠) 인식형 Safety Shield를 연동했습니다.

### 6.2 객체 검출 정확도 향상 (`YOLO11l @ 1280` + 중복 제거 병합 + Multi-Scale TTA)
- **데이터 무결성 확장**: `leducnhuan/kitti-tracking` + `harshpdani/kitti-stereo-image`(`7,481`장 중 Val/Test `0002,0012,0015,0017,0013,0016,0019`와 MD5/dHash 일치 `595`장 엄격 제거 → 고유 `6,886`장 추가, 총 `14,371` 학습 엔트리) + `Van`/`Person_sitting` 114 그레이 마스킹.
- **정량 비교 (`v2 → v4`, [`eval_metrics_v4.json`](./results/v4_road_lane/detect_outputs/outputs/eval_metrics_v4.json))**:
  - **Val (`All` 난이도)**: `v2 mAP@0.5 = 0.5235` → `v4 std = 0.6889` (`Precision 0.8685`) → **`v4 TTA = 0.7400`** (`Car 0.8475 / Ped 0.7712 / Cyc 0.6014`).
  - **Test (`All` 난이도)**: `v2 mAP@0.5 = 0.4269` → **`v4 TTA = 0.6705`** (`Precision 0.7984`, `Car 0.7804 / Ped 0.6441 / Cyc 0.5870`).
  - **KITTI 표준 난이도별 평가 (TTA)**: Val **Easy `mAP@0.5 = 0.8677`** (`Precision 0.9750`), Test **Easy `mAP@0.5 = 0.8966`** (`~0.90`, `Car 0.9792 / Ped 0.9206 / Cyc 0.7899`), Test **Moderate `mAP@0.5 = 0.8404`**, Test **Hard `mAP@0.5 = 0.8025`**, Test **KITTI-All `mAP@0.5 = 0.7965`** (`mean_precision 0.8531`).
- **v4 World Model 최종 시연 및 RQ1–RQ4 재검증**:
  - Val `0015` ([`full MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_full/world_model_demo_0015.mp4) / [`staged MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_staged/world_model_demo_0015.mp4)): `ADE 3.5836 m`, RQ3 `0.2277 < 0.2838 < 0.2916`, RQ4 스파이크 `8 → 0 (-100%)`.
  - Test `0019` ([`full MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_full/world_model_demo_0019.mp4) / [`staged MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_staged/world_model_demo_0019.mp4)): `ADE 2.0437 m`, RQ3 `0.5421 < 0.6261 < 0.6454`, RQ4 스파이크 `16 → 0 (-100%)`, 플립 `124 → 53 (-57.26%)`.
