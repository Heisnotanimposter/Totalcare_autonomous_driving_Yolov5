# Dataset Audit: KITTI Tracking (`image_02` & `label_02`)

## 1. Dataset Structure

The perception and temporal world model pipeline uses the **KITTI Object Tracking** training benchmark:
- **Images**: `training/image_02/<seq>/<frame>.png` (`data_tracking_image_2.zip`, `14.73 GiB`, `10 Hz` sequential RGB frames)
- **Labels**: `training/label_02/<seq>.txt` (`data_tracking_label_2.zip`, `2.15 MiB`, 17-column format)
- **Calibration**: `training/calib/<seq>.txt` (`data_tracking_calib.zip`, containing `P2` camera projection matrix for ground-plane ego transformation)

Canonical Google Drive directory preserved from Colab workflow ([untitled63.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/untitled63.py), [01_kitti_colab_audit_and_prep.ipynb](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/colab/01_kitti_colab_audit_and_prep.ipynb)):
- `/content/drive/MyDrive/KITTI_Tracking/`

---

## 2. Integrity Verification Protocol

Implemented in [src/kitti_prep.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/kitti_prep.py) (`audit_images_and_labels`) and initially audited in [untitled63.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/untitled63.py):

1. **Archive CRC & Decode Check**:
   - `ZipFile.testzip()` verifies zero corrupt archive members.
   - `PIL.Image.verify()` checks image readability across all `8,008` training frames across `21` sequences (`0000`–`0020`).
2. **1:1 Image-Label Correspondence**:
   - Every sequence ID in `training/image_02/` matches `training/label_02/`.
   - Counts `missing_image_count`, `extra_image_count`, `duplicate_image_frame_count`, and `label_frame_without_image_count`.
   - Frames with zero annotated objects (empty label frames) are retained as valid negative frames in YOLO export.
3. **Label Row Validation**:
   - Enforces 17 fields per row, finite numeric values, and `x2 > x1`, `y2 > y1`.
4. **Sequence-Scoped Track IDs**:
   - KITTI `track_id` values reset per sequence and are always treated as composite `(sequence_id, track_id)`.
5. **DontCare & Distractor Masking**:
   - Target classes: `Car`, `Pedestrian`, `Cyclist`.
   - Adjacent KITTI classes (`Van` for `Car`; `Person`, `Person_sitting` for `Pedestrian`) and `DontCare` regions are masked with neutral pixel value `114` during YOLO export (`pixel_mask_policy = dontcare_and_distractors`), preventing false negative penalties against TrackEval distractor taxonomy.

---

## 3. Sequence-Disjoint Data Split

To prevent temporal data leakage, frames from the same sequence are **never** split across Train, Validation, and Test sets (`validate_disjoint_sequences` in [src/kitti_prep.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/kitti_prep.py)):

| Split | Sequence Count | Sequence IDs | Purpose |
|---|---:|---|---|
| **Train** | 14 | `0000, 0001, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0010, 0011, 0014, 0018, 0020` | Detector training (`Car`, `Pedestrian`, `Cyclist`) |
| **Val** | 4 | `0002, 0012, 0015, 0017` | Epoch validation & tracking tuning (`0015`) |
| **Test** | 3 | `0013, 0016, 0019` | Held-out generalization & world-model demo (`0019`) |

Exact per-sequence frame counts, per-split class distributions, and empty-frame ratios are generated automatically by `kitti_prep.py` into `results/kitti_step2_audit.json` and `results/kitti_export_manifest.json` during remote Colab/Kaggle execution.

