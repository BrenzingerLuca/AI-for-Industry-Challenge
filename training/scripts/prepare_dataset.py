"""
prepare_dataset.py - YOLOv8-Pose Dataset Preparation Script
--------------------------------------
This script performs the following tasks:
1. Loads raw images and labels from the global 'data/datasets' directory.
2. Splits the data into 'train' and 'val' sets based on a configurable ratio.
3. Fixes out-of-bounds coordinates (clipping 0.0-1.0) to prevent YOLO training errors.
4. Organizes the split into a structured folder for training.
5. Generates a ZIP file ready for Google Colab upload.

run the script from inside ~/ws_aic/src/aic/aic_solution/training/ by using python3 scripts/prepare_dataset.py
"""

import sys
import os
import shutil
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The name of the folder inside data/datasets/
RAW_DATA_NAME = "single_sc_port_dataset"

# The name the processed dataset is saved
DATASET_NAME = "single_sc_port" 

# Train/Val split ratio (0.2 means 20% for validation)
VAL_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42
# ==============================================================================

def clean_and_fix_labels(label_path):
    """
    Ensures all YOLO coordinates are within the valid [0.0, 1.0] range.
    Fixes 'corrupt label' errors in YOLOv8 training.
    """
    if not os.path.exists(label_path):
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if not parts:
            continue
        
        class_id = int(parts[0])
        # Clip coordinates (BBox and Keypoints) to be exactly between 0 and 1
        coords = np.clip(parts[1:], 0.0, 1.0)
        
        fixed_line = f"{class_id} " + " ".join([f"{c:.6f}" for c in coords])
        fixed_lines.append(fixed_line)
    
    with open(label_path, 'w') as f:
        f.write("\n".join(fixed_lines))

def main():
    # Define absolute-style paths relative to this script's location
    # Assuming script is run from inside ~/ws_aic/src/aic/aic_solution/training/
    base_dir = os.getcwd()
    source_dir = os.path.join(base_dir, "../data/datasets", RAW_DATA_NAME)
    target_base = os.path.join(base_dir, "prepared_datasets")
    split_dir = os.path.join(target_base, f"{DATASET_NAME}_split")
    zip_path = os.path.join(target_base, f"{DATASET_NAME}_colab.zip")

    print(f"--- Preparing Dataset: {DATASET_NAME} ---")

    if os.path.basename(base_dir) != "training":
        print("\nTo ensure paths are resolved correctly, please run this script")
        print("from the 'training' directory: Using python3 scripts/prepare_dataset.py")
        sys.exit(1)
        
    # 1. Validation of source paths
    src_images = os.path.join(source_dir, "images")
    src_labels = os.path.join(source_dir, "labels")

    if not os.path.exists(src_images):
        print(f"Error: Source images not found at {src_images}")
        return

    # 2. Cleanup previous split
    if os.path.exists(split_dir):
        print(f"Removing existing split folder: {split_dir}")
        shutil.rmtree(split_dir)
    
    os.makedirs(target_base, exist_ok=True)

    # 3. Perform Train/Test Split
    all_images = sorted([f for f in os.listdir(src_images) if f.endswith('.jpg')])
    train_imgs, val_imgs = train_test_split(
        all_images, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    print(f"Statistics: {len(train_imgs)} train, {len(val_imgs)} val images.")

    # 4. Copy and process files
    for split_name, files in [('train', train_imgs), ('val', val_imgs)]:
        print(f"Processing '{split_name}' set...")
        
        dest_img_dir = os.path.join(split_dir, split_name, "images")
        dest_lab_dir = os.path.join(split_dir, split_name, "labels")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lab_dir, exist_ok=True)

        for f in files:
            # Copy Image
            shutil.copy(os.path.join(src_images, f), os.path.join(dest_img_dir, f))
            
            # Copy and Fix Label
            label_name = f.replace('.jpg', '.txt')
            src_label_path = os.path.join(src_labels, label_name)
            dest_label_path = os.path.join(dest_lab_dir, label_name)
            
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
                clean_and_fix_labels(dest_label_path)

    # 5. Create ZIP for Colab
    print(f"Zipping dataset to: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, filenames in os.walk(split_dir):
            for f in filenames:
                full_path = os.path.join(root, f)
                # Keep the 'train/...' and 'val/...' structure inside the zip
                relative_path = os.path.relpath(full_path, split_dir)
                z.write(full_path, relative_path)

    print(f"\n Success!")
    print(f"Final ZIP: {zip_path}")
    print(f"Split Folder: {split_dir}")

if __name__ == "__main__":
    main()