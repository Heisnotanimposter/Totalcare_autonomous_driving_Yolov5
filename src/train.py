#!/usr/bin/env python3
"""
Model Training Script for Total Care Autonomous Driving System
Supports YOLOv5 (s, m, l, x) and YOLOv9 (c, e) model training with custom hyperparameter logging.
"""

import argparse
import sys
import os
def check_dependencies():
    try:
        import ultralytics
        import torch
        import yaml
        print(f"[+] PyTorch Version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        print(f"[+] Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        print("[!] Ultralytics, PyTorch, or PyYAML not found. Installing dependencies...")
        os.system(f"{sys.executable} -m pip install -r requirements.txt")


def run_training(model_name: str, data_config: str, epochs: int, batch_size: int, img_size: int, project: str, run_name: str, device: str):
    """
    Executes YOLO model training using the Ultralytics API.
    """
    from ultralytics import YOLO

    print("\n=======================================================")
    print(f"   TOTAL CARE AUTONOMOUS DRIVING MODEL TRAINING")
    print(f"   Model: {model_name} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"   Image Size: {img_size} | Config: {data_config}")
    print("=======================================================\n")

    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset configuration file not found at: {data_config}")

    # Initialize model (downloads pretrained weights automatically if not locally cached)
    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=run_name,
        device=device,
        exist_ok=True,
        plots=True
    )

    print(f"\n[✓] Training Complete! Logs & Weights saved in: {os.path.join(project, run_name)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Total Care YOLO Training Script")
    parser.add_argument("--model", type=str, default="yolov5s.pt", help="Model weights or cfg (e.g. yolov5s.pt, yolov5m.pt, yolov9c.pt)")
    parser.add_argument("--data", type=str, default="./data_config.yaml", help="Path to dataset configuration YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size (pixels)")
    parser.add_argument("--project", type=str, default="./runs/train", help="Save directory project name")
    parser.add_argument("--name", type=str, default="total_care_exp", help="Save run name")
    parser.add_argument("--device", type=str, default="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", help="Device to run on (e.g. 0, cpu, mps)")

    args = parser.parse_args()

    check_dependencies()
    run_training(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project=args.project,
        run_name=args.name,
        device=args.device
    )


if __name__ == "__main__":
    main()
