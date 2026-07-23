#!/usr/bin/env python3
"""
Model Export Script for Autonomous Driving Embedded Platforms
Exports trained YOLO PyTorch (.pt) weights to ONNX, TorchScript, TensorRT, or CoreML.
"""

import argparse
import os


def export_model(weights_path: str, format_type: str, imgsz: int):
    """
    Exports YOLO model using the Ultralytics export pipeline.
    """
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"[+] Loading model weights: {weights_path}")
    model = YOLO(weights_path)

    print(f"[+] Exporting model to format: {format_type} (Image size: {imgsz})")
    exported_path = model.export(format=format_type, imgsz=imgsz)

    print(f"[✓] Model successfully exported to: {exported_path}")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Export Trained YOLO Weights for Autonomous Embedded Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained PyTorch weights file (.pt)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript", "coreml", "tflite"], help="Export format (onnx, engine for TensorRT, torchscript, coreml, tflite)")
    parser.add_argument("--imgsz", type=int, default=416, help="Export image resolution size")

    args = parser.parse_args()
    export_model(args.weights, args.format, args.imgsz)


if __name__ == "__main__":
    main()
