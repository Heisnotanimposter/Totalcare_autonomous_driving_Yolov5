#!/usr/bin/env python3
"""
Real-time Autonomous Driving Detection & Hazard Alert Inference Engine
Runs object detection on images, directories, or video streams and visualizes safety hazard alerts.
"""

import argparse
import os
import sys
import glob


HAZARD_RULES = {
    "pothole": {"color": (0, 0, 255), "alert": "[CRITICAL] ROAD HAZARD: POTHOLE DETECTED"},
    "no_belt": {"color": (0, 0, 255), "alert": "[SAFETY ALERT] UNBELTED OCCUPANT DETECTED"},
    "deer": {"color": (0, 165, 255), "alert": "[COLLISION RISK] WILDLIFE DETECTED"},
    "roe_deer": {"color": (0, 165, 255), "alert": "[COLLISION RISK] WILDLIFE DETECTED"},
    "elk": {"color": (0, 165, 255), "alert": "[COLLISION RISK] WILDLIFE DETECTED"},
    "waterdeer": {"color": (0, 165, 255), "alert": "[COLLISION RISK] WILDLIFE DETECTED"},
    "trafficLight-Red": {"color": (0, 0, 255), "alert": "[TRAFFIC CONTROL] RED LIGHT AHEAD"},
    "pedestrian": {"color": (0, 255, 255), "alert": "[VRU CAUTION] PEDESTRIAN NEARBY"},
    "biker": {"color": (0, 255, 255), "alert": "[VRU CAUTION] CYCLIST NEARBY"},
    "scooter": {"color": (0, 255, 255), "alert": "[VRU CAUTION] SCOOTER NEARBY"},
}


def process_image(model, img_path: str, conf_thresh: float, save_dir: str, show_gui: bool):
    """
    Runs inference on a single image and overlays hazard alerts.
    """
    import cv2
    import numpy as np

    results = model(img_path, conf=conf_thresh)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Could not load image: {img_path}")
        return

    active_alerts = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            color = (0, 255, 0)
            if cls_name in HAZARD_RULES:
                color = HAZARD_RULES[cls_name]["color"]
                active_alerts.append(HAZARD_RULES[cls_name]["alert"])

            # Draw bounding box
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)

            # Draw label tag
            label_text = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (xyxy[0], xyxy[1] - 20), (xyxy[0] + w, xyxy[1]), color, -1)
            cv2.putText(img, label_text, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw active hazard alert banner on top of image
    if active_alerts:
        unique_alerts = list(set(active_alerts))
        banner_height = 30 * len(unique_alerts) + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], banner_height), (0, 0, 128), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        for idx, alert_msg in enumerate(unique_alerts):
            cv2.putText(img, alert_msg, (10, 25 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"[✓] Processed: {img_path} -> Saved: {out_path} (Detections: {len(results[0].boxes)})")

    if show_gui:
        cv2.imshow("Total Care Autonomous Driving Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_inference(weights: str, source: str, conf_thresh: float, save_dir: str, show_gui: bool):
    """
    Main inference loop for single image, directory, or video.
    """
    import cv2
    from ultralytics import YOLO

    print(f"[+] Loading model weights from: {weights}")
    model = YOLO(weights)

    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"[+] Running video inference on: {source}")
            cap = cv2.VideoCapture(source)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(save_dir, exist_ok=True)
            out_video = os.path.join(save_dir, "output_" + os.path.basename(source))
            writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=conf_thresh)
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

            cap.release()
            writer.release()
            print(f"[✓] Processed video saved to: {out_video}")
        else:
            process_image(model, source, conf_thresh, save_dir, show_gui)
    elif os.path.isdir(source):
        images = glob.glob(os.path.join(source, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(source, "*.[pP][nN][gG]"))
        print(f"[+] Found {len(images)} images in directory: {source}")
        for img_path in images:
            process_image(model, img_path, conf_thresh, save_dir, show_gui=False)


def main():
    parser = argparse.ArgumentParser(description="Total Care Object Detection & Hazard Alert Script")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="Model weights path (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image, image directory, or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-dir", type=str, default="./runs/detect", help="Output directory for annotated results")
    parser.add_argument("--show", action="store_true", help="Display GUI window with detected bounding boxes")

    args = parser.parse_args()
    run_inference(args.weights, args.source, args.conf, args.save_dir, args.show)


if __name__ == "__main__":
    main()
