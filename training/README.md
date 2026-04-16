open colab with the following link: https://colab.research.google.com/

## What the Losses Mean

**box_loss — 0.378**
How accurately the rectangular bounding box sits around the port.
Measures center point, width, and height. At 0.378 the box is very precise.

**pose_loss — 0.168**
How accurately the 4 keypoints (the corners of the port) are predicted.
This is the most important value for us — the keypoints are needed to determine
the orientation and exact position of the port. At 0.168 this is very good.

**cls_loss — 0.261**
How confidently the model distinguishes between `sfp_port_0` and `sfp_port_1`.
At 0.261 the model makes almost no mistakes between the two ports.

---

Simplified:
- **box_loss** = "Can I find the port?"
- **pose_loss** = "Can I detect the exact corners and orientation?"
- **cls_loss** = "Do I know which port it is?"

---

## How the Numbers Are Calculated

All three losses follow the same principle: **prediction vs. ground truth**,
and the difference is expressed as a number. Lower = better, 0 would be perfect.

---

### box_loss — CIoU Loss

Measures how much the predicted box deviates from the ground truth.
CIoU (Complete Intersection over Union) computes three things simultaneously:


CIoU = 1 - IoU + center point distance + aspect ratio penalty
IoU is the overlap ratio:

      ┌─────────────┐
      │  predicted  │
      │    ┌────────┼────┐
      │    │  IoU   │    │
      └────┼────────┘    │
           │   ground    │
           │   truth     │
           └─────────────┘

IoU = 1.0 → perfect overlap → box_loss = 0

---

### pose_loss — OKS (Object Keypoint Similarity)

For each of the 4 keypoints, the Euclidean distance between the predicted
point and the ground truth point is measured, normalized by object size:
OKS = exp( -d² / (2 * s² * σ²) )
d = distance between predicted and ground truth keypoint (in pixels)
s = object size (square root of bounding box area)
σ = per-keypoint constant

OKS = 1.0 → keypoint sits perfectly → pose_loss = 0

At a pose_loss of 0.168, the predicted corners are on average
only a few pixels off from the ground truth.

---

### cls_loss — Binary Cross Entropy

For each class the model outputs a probability, e.g.:
Predicted:    sfp_port_0 = 0.97,  sfp_port_1 = 0.03
Ground truth: sfp_port_0 = 1.0,   sfp_port_1 = 0.0

Cross Entropy penalizes wrong probabilities logarithmically:
Loss = -log(0.97) = 0.03   ← almost correct, small loss
Loss = -log(0.03) = 3.51   ← wrong, large loss
At 0.261 the model is almost always very confident about the correct class.