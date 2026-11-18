# Results Directory

Stores quantitative evaluation outputs, rendered split-screen demonstration videos (`>= 5s` @ `10 Hz`), trajectory exports, and keyframe visualizations downloaded from Kaggle CLI / Colab CLI runs.

Expected artifacts after run completion:
- `kitti_step2_audit.json`
- `kitti_export_manifest.json`
- `eval_metrics.json` (Detection `Precision`, `Recall`, `mAP@0.5`, `mAP@0.5:0.95` on Val/Test)
- `tracking_summary.json` & `track_<seq>/trajectories.csv`, `kitti_preds/<seq>.txt`
- `world_model_eval_<seq>.json` (Depth `Abs Rel`, `RMSE`; Motion `ADE`, `FDE`)
- `world_model_demo_<seq>.mp4` & `keyframes/*.png`

