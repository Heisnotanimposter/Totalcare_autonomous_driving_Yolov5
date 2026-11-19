# 데이터셋 무결성 감사: KITTI Tracking (`image_02` 및 `label_02`)

## 1. 데이터셋 구조

본 인식 및 시계열 World Model 파이프라인은 **KITTI Object Tracking** 학습 벤치마크를 사용합니다:
- **이미지**: `training/image_02/<seq>/<frame>.png` (`data_tracking_image_2.zip`, `14.73 GiB`, `10 Hz` 연속 RGB 프레임)
- **라벨**: `training/label_02/<seq>.txt` (`data_tracking_label_2.zip`, `2.15 MiB`, 17-column 형식)
- **캘리브레이션**: `training/calib/<seq>.txt` (`data_tracking_calib.zip`, 지면 자차 좌표 변환용 `P2` 카메라 투영 행렬 포함)

Colab 워크플로우([untitled63.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/untitled63.py), [01_kitti_colab_audit_and_prep.ipynb](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/colab/01_kitti_colab_audit_and_prep.ipynb))에서 유지되는 표준 Google Drive 경로:
- `/content/drive/MyDrive/KITTI_Tracking/`

---

## 2. 무결성 검증 프로토콜

[src/kitti_prep.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/kitti_prep.py) (`audit_images_and_labels`)에 구현되고 [untitled63.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/untitled63.py)에서 초기 감사를 수행했습니다:

1. **아카이브 CRC 및 이미지 디코드 검증**:
   - `ZipFile.testzip()`으로 손상된 압축 항목이 0건임을 확인합니다.
   - `PIL.Image.verify()`로 `21`개 시퀀스(`0000`–`0020`) 전체 `8,008`장 학습 프레임의 이미지 가독성을 전수 검사합니다.
2. **이미지-라벨 1:1 정합성 검사**:
   - `training/image_02/`의 모든 시퀀스 ID가 `training/label_02/`와 일치하는지 확인합니다.
   - `missing_image_count`, `extra_image_count`, `duplicate_image_frame_count`, `label_frame_without_image_count`를 집계합니다.
   - 주석 객체가 0개인 프레임(빈 라벨 프레임)도 YOLO 변환 시 유효한 음성(Negative) 프레임으로 보존합니다.
3. **라벨 행(Row) 유효성 검증**:
   - 행당 17개 필드, 유한 실수값, `x2 > x1`, `y2 > y1` 조건을 강제합니다.
4. **시퀀스 단위 Track ID 격리**:
   - KITTI `track_id`는 시퀀스마다 0부터 초기화되므로 항상 복합 키 `(sequence_id, track_id)`로 처리합니다.
5. **DontCare 및 Distractor 마스킹**:
   - 타겟 부류: `Car`, `Pedestrian`, `Cyclist`.
   - 인접 혼동 부류(`Car`의 인접 부류 `Van`, `Pedestrian`의 인접 부류 `Person`, `Person_sitting`)와 `DontCare` 영역을 YOLO 변환 시 중립 회색 픽셀 `114`로 마스킹(`pixel_mask_policy = dontcare_and_distractors`)하여 TrackEval 평가 기준 대비 불필요한 False-Negative 페널티를 방지합니다.

---

## 3. 시퀀스 비독립(Sequence-Disjoint) 데이터 분할

시계열 데이터 누수(Data Leakage)를 원천 차단하기 위해 동일 시퀀스의 프레임은 Train, Validation, Test 세트 간에 **절대** 분할하지 않습니다([src/kitti_prep.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/kitti_prep.py)의 `validate_disjoint_sequences`):

| 분할(Split) | 시퀀스 수 | 시퀀스 ID | 용도 |
|---|---:|---|---|
| **Train** | 14 | `0000, 0001, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0010, 0011, 0014, 0018, 0020` | 검출기 학습 (`Car`, `Pedestrian`, `Cyclist`) |
| **Val** | 4 | `0002, 0012, 0015, 0017` | Epoch별 검증 및 추적 튜닝 (`0015`) |
| **Test** | 3 | `0013, 0016, 0019` | 완전 미학습 일반화 평가 및 World Model 시연 (`0019`) |

시퀀스별 프레임 수, 분할별 부류 분포, 빈 프레임 비율은 원격 Colab/Kaggle 실행 시 `kitti_prep.py`에 의해 `results/kitti_step2_audit.json` 및 `results/kitti_export_manifest.json`으로 자동 생성됩니다.
