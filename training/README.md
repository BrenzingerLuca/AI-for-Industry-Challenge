# Training — Port Keypoint Models

This folder contains two **separate** YOLOv8n-pose models. Each one detects a port
and predicts its **4 corner keypoints**, which are used downstream to estimate the
port's position and orientation.

| Model | File | Classes | Used by |
|---|---|---|---|
| **SFP port model** (NIC card) | `models/best150.pt` | `sfp_port_0`, `sfp_port_1` | `testing/keypoint_estimator_node.py`, `pose_estimation/multi_cam_triangulation_debug_node.py` |
| **SC port model** | `models/single_sc_detection.pt` | `sc_port` | `pose_estimation/pose_estimation.py`, `pose_estimation/capture_keypoints.py`, `aic_solution_policy/.../qualification/config.py` |

`models/best.pt` is an earlier 50-epoch run of the SFP model and has been superseded by `best150.pt`.

---

## Results at a glance

Validation metrics recorded by Ultralytics at the end of each training run (standard settings, `conf=0.001`, `IoU=0.7`).

| Model | Box mAP50 | Box mAP50-95 | **Pose mAP50** | **Pose mAP50-95** | Precision | Recall |
|---|---|---|---|---|---|---|
| SFP port (`best150.pt`) | 0.928 | 0.841 | **0.933** | **0.919** | 0.995 | 0.926 |
| SC port (`single_sc_detection.pt`) | 0.975 | 0.914 | **0.985** | **0.976** | 1.000 | 0.970 |
| *SFP port, 50 epochs (`best.pt`, superseded)* | *0.929* | *0.819* | *0.932* | *0.910* | *0.986* | *0.926* |

**Pose mAP50-95 is the headline metric** for both models, since it measures how precisely the
corner keypoints are placed — which is what the pose estimation depends on.

---

## Model 1 — SFP port model (`best150.pt`)

Detects the two SFP ports on the NIC card and distinguishes them (`sfp_port_0` / `sfp_port_1`).

**Dataset** — `prepared_datasets/single_nic_card_split` (`single_nic_card_colab.zip`)

| Split | Images | Port instances |
|---|---|---|
| train | 324 | 636 |
| val | 82 | 161 |

**Training setup**

| Parameter | Value |
|---|---|
| Base model | `yolov8n-pose.pt` |
| Epochs | 150 (all epochs ran; best checkpoint around epoch 139) |
| Image size | 640 |
| Batch size | 16 |
| Patience | 20 |
| Optimizer / lr0 | auto / 0.01 |
| Keypoints | `kpt_shape: [4, 3]` — TL, TR, BR, BL |

**Validation results**

| Metric | Box | Pose (keypoints) |
|---|---|---|
| Precision | 0.995 | 0.995 |
| Recall | 0.926 | 0.926 |
| mAP50 | 0.928 | 0.933 |
| mAP50-95 | 0.841 | 0.919 |

Compared to the 50-epoch run (`best.pt`), training for 150 epochs mainly improved localisation
precision (Box mAP50-95 0.819 → 0.841, Pose mAP50-95 0.910 → 0.919); mAP50 is essentially unchanged.
Recall (≈0.93) is the limiting factor: a few ports in the validation set are not detected at all.

---

## Model 2 — SC port model (`single_sc_detection.pt`)

Detects a single SC port (one class, `sc_port`).

**Dataset** — `prepared_datasets/single_sc_port_split` (`single_sc_port_colab.zip`)

| Split | Images | Port instances |
|---|---|---|
| train | 268 | 268 |
| val | 68 | 68 |

**Training setup**

| Parameter | Value |
|---|---|
| Base model | `yolov8n-pose.pt` |
| Epochs | 150 (all epochs ran; best checkpoint at the final epoch) |
| Image size | 640 |
| Batch size | 16 |
| Patience | 30 |
| Optimizer / lr0 | auto / 0.01 |
| Keypoints | `kpt_shape: [4, 3]` — 4 port corners |

**Validation results**

| Metric | Box | Pose (keypoints) |
|---|---|---|
| Precision | 1.000 | 1.000 |
| Recall | 0.970 | 0.970 |
| mAP50 | 0.975 | 0.985 |
| mAP50-95 | 0.914 | 0.976 |

The best checkpoint was the last epoch, so the model was still improving when training stopped —
more epochs may give a small further gain.

---

## What the metrics mean

**Precision** — of all ports the model predicted, how many were real. **Recall** — of all real ports,
how many the model found.

**mAP (mean Average Precision)** summarises precision and recall over all confidence thresholds into
one number between 0 and 1 (higher = better), averaged over the classes.

- **Box mAP** counts a prediction as correct when its bounding box overlaps the ground truth by at
  least a given **IoU** (Intersection over Union).
- **Pose mAP** uses **OKS** (Object Keypoint Similarity) instead of IoU: how close the 4 predicted
  corners are to the ground-truth corners, normalised by object size.

  ```
  OKS = mean over keypoints of exp( -d² / (2 · s² · σ²) )
  d = pixel distance predicted ↔ ground-truth keypoint
  s = object scale (from the box area)
  σ = per-keypoint tolerance constant
  ```

The suffix gives the threshold:

- **mAP50** — a match needs IoU/OKS ≥ 0.50. Answers *"does the model find the port?"*
- **mAP50-95** — the average over thresholds 0.50, 0.55, …, 0.95. Only high if predictions are also
  **precise**, so it's the stricter and more meaningful number for us.

---

## Workflow

1. **Prepare the dataset** — set `RAW_DATA_NAME` and `DATASET_NAME` in `scripts/prepare_dataset.py`,
   then run from `~/ws_aic/src/aic/aic_solution/training/`:
   ```bash
   python3 scripts/prepare_dataset.py
   ```
   It splits the data 80/20 into train/val (seed 42), clips out-of-bounds coordinates to [0, 1] and
   writes `prepared_datasets/<name>_split/` plus `<name>_colab.zip`.

2. **Train in Colab** — open `notebooks/train_YOLOv8_pose_colab.ipynb` in
   [Google Colab](https://colab.research.google.com/) with a T4 GPU runtime, upload the zip and run
   the cells. Adjust the config cell to the model you're training:
   - SFP model: `names: {0: sfp_port_0, 1: sfp_port_1}`
   - SC model: `names: {0: sc_port}`

   Set `EPOCHS`, `PATIENCE` and `EXPERIMENT_NAME` accordingly (see the tables above), then download
   `best.pt` and save it under `models/` with a descriptive name.

3. **Evaluate** — `notebooks/evaluation.ipynb` computes Box/Pose mAP50 and mAP50-95 per class and
   shows sample predictions. Run `model.val()` with the default confidence threshold (0.001) when
   reporting mAP; a higher threshold (e.g. 0.25) is fine for visual inspection but distorts the mAP
   values.
