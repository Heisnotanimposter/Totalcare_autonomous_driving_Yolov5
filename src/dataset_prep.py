#!/usr/bin/env python3
"""
Dataset Preparation & Remapping Tool for Autonomous Driving YOLO System
Supports splitting images/labels, remapping class IDs across multi-source datasets,
and writing YOLO-formatted data configuration YAML files.
"""

import argparse
import os
import shutil
from glob import glob


def remap_labels_in_file(filepath: str, old_to_new_map: dict, default_class: int = None):
    """
    Remaps class IDs in a YOLO format annotation file.
    """
    if not os.path.exists(filepath):
        return

    updated_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
            if cls_id in old_to_new_map:
                parts[0] = str(old_to_new_map[cls_id])
            elif default_class is not None:
                parts[0] = str(default_class)
            updated_lines.append(' '.join(parts) + '\n')
        except ValueError:
            continue

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)


def prepare_yolo_dataset(raw_dir: str, output_dir: str, val_split: float = 0.2, test_split: float = 0.1):
    """
    Splits images and corresponding label text files into standard YOLO dataset structure:
      output_dir/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
    """
    img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob(os.path.join(raw_dir, '**', ext), recursive=True))

    if not all_images:
        print(f"[!] No images found in directory: {raw_dir}")
        return

    print(f"[+] Total images discovered: {len(all_images)}")

    # Pair with matching label files
    valid_pairs = []
    for img_path in all_images:
        base, _ = os.path.splitext(img_path)
        # Check potential label paths
        label_path = base + '.txt'
        if not os.path.exists(label_path):
            # Check labels folder sibling
            parts = img_path.split(os.sep)
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'labels'
                label_path = os.path.splitext(os.sep.join(parts))[0] + '.txt'

        if os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))

    print(f"[+] Total valid (image, label) pairs: {len(valid_pairs)}")
    if not valid_pairs:
        print("[!] No matching annotation files (.txt) found.")
        return

    # Train / Val / Test split
    from sklearn.model_selection import train_test_split
    train_val, test_data = train_test_split(valid_pairs, test_size=test_split, random_state=42)
    relative_val_size = val_split / (1.0 - test_split)
    train_data, val_data = train_test_split(train_val, test_size=relative_val_size, random_state=42)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, pairs in splits.items():
        img_out = os.path.join(output_dir, 'images', split_name)
        lbl_out = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_p, lbl_p in pairs:
            shutil.copy(img_p, os.path.join(img_out, os.path.basename(img_p)))
            shutil.copy(lbl_p, os.path.join(lbl_out, os.path.basename(lbl_p)))

        print(f"[✓] Copy finished for split '{split_name}': {len(pairs)} samples")


def create_yaml_config(output_yaml: str, dataset_root: str, class_names: list):
    """
    Generates data.yaml file required for Ultralytics YOLO training.
    """
    import yaml
    config = {
        'path': os.path.abspath(dataset_root),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[✓] Created dataset YAML configuration: {output_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation & Splitting Tool")
    parser.add_argument("--raw-dir", type=str, default="./raw_dataset", help="Directory containing unorganized dataset")
    parser.add_argument("--output-dir", type=str, default="./dataset", help="Output dataset directory structure")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation dataset ratio (default: 0.2)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test dataset ratio (default: 0.1)")
    parser.add_argument("--config-out", type=str, default="./data_config.yaml", help="Path to save output yaml config")

    args = parser.parse_args()

    default_classes = [
        'biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green',
        'trafficLight-GreenLeft', 'trafficLight-Red', 'deer', 'roe_deer',
        'elk', 'waterdeer', 'pothole', 'scooter', 'belt', 'no_belt',
        'blocked_left', 'blocked_right', 'blocked_straight', 'free_path'
    ]

    print("=== Total Care Dataset Preparation ===")
    prepare_yolo_dataset(args.raw_dir, args.output_dir, args.val_split, args.test_split)
    create_yaml_config(args.config_out, args.output_dir, default_classes)


if __name__ == "__main__":
    main()
