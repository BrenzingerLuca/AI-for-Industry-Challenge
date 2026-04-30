#!/usr/bin/env python3
"""Interactive image labeler for YOLO keypoint/pose format.

Writes one YOLO-pose label line per image:
  <class> <cx> <cy> <w> <h> <kp1x> <kp1y> <v1> ...

All values are normalized to [0, 1]. Visibility flag v is an integer:
  0 = not labeled, 1 = labeled but not visible, 2 = visible

This tool is intentionally minimal and matches the folder layout already used in
this repo:
  <dataset>/images/*.jpg
  <dataset>/labels/*.txt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class Annotation:
	class_id: int = 0
	# bbox in pixel coords: (x1, y1, x2, y2) with x1<x2, y1<y2
	bbox: Optional[tuple[int, int, int, int]] = None
	# keypoints in pixel coords: [(x, y, v), ...]
	keypoints: list[tuple[int, int, int]] = None  # type: ignore[assignment]

	def __post_init__(self) -> None:
		if self.keypoints is None:
			self.keypoints = []


def _clamp(v: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, v))


def _sorted_image_files(images_dir: Path) -> list[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp"}
	files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
	files.sort()
	return files


def _label_path(labels_dir: Path, image_path: Path) -> Path:
	return labels_dir / (image_path.stem + ".txt")


def _parse_existing_label(label_file: Path, num_keypoints: int) -> Optional[Annotation]:
	if not label_file.exists():
		return None

	try:
		raw = label_file.read_text(encoding="utf-8").strip()
	except Exception:
		return None

	if not raw:
		return None

	# Only support 1 object per image for this simple tool.
	first_line = raw.splitlines()[0].strip()
	parts = first_line.split()
	expected = 5 + 3 * num_keypoints
	if len(parts) != expected:
		return None

	try:
		class_id = int(float(parts[0]))
		floats = [float(x) for x in parts[1:]]
	except Exception:
		return None

	# We cannot reconstruct bbox pixel coords without image size here.
	# We'll store normalized values temporarily in a special way by returning
	# bbox/keypoints as empty and letting the caller re-hydrate with image dims.
	ann = Annotation(class_id=class_id, bbox=None, keypoints=[])
	ann._normalized = floats  # type: ignore[attr-defined]
	return ann


def _hydrate_from_normalized(ann: Annotation, img_w: int, img_h: int, num_keypoints: int) -> Annotation:
	floats = getattr(ann, "_normalized", None)
	if floats is None:
		return ann

	cx, cy, bw, bh = floats[0:4]
	x1 = int(round((cx - bw / 2.0) * img_w))
	y1 = int(round((cy - bh / 2.0) * img_h))
	x2 = int(round((cx + bw / 2.0) * img_w))
	y2 = int(round((cy + bh / 2.0) * img_h))
	ann.bbox = (x1, y1, x2, y2)

	kpts = []
	kp_floats = floats[4:]
	for i in range(num_keypoints):
		kx = kp_floats[i * 3 + 0]
		ky = kp_floats[i * 3 + 1]
		v = int(round(kp_floats[i * 3 + 2]))
		kpts.append((int(round(kx * img_w)), int(round(ky * img_h)), v))
	ann.keypoints = kpts

	try:
		delattr(ann, "_normalized")
	except Exception:
		pass
	return ann


def _ann_to_yolo_line(ann: Annotation, img_w: int, img_h: int, num_keypoints: int, bbox_mode: str,
					  auto_box_size_px: int) -> str:
	if len(ann.keypoints) != num_keypoints:
		raise ValueError(f"Need exactly {num_keypoints} keypoints, got {len(ann.keypoints)}")

	if bbox_mode == "draw":
		if ann.bbox is None:
			raise ValueError("Missing bounding box (draw mode)")
		x1, y1, x2, y2 = ann.bbox
	else:
		# auto bbox around keypoints
		xs = [kp[0] for kp in ann.keypoints]
		ys = [kp[1] for kp in ann.keypoints]
		min_x, max_x = min(xs), max(xs)
		min_y, max_y = min(ys), max(ys)
		# If only one point, create a square box around it.
		if min_x == max_x and min_y == max_y:
			half = max(2, int(auto_box_size_px // 2))
			x1, y1, x2, y2 = min_x - half, min_y - half, max_x + half, max_y + half
		else:
			pad = max(2, int(auto_box_size_px // 2))
			x1, y1, x2, y2 = min_x - pad, min_y - pad, max_x + pad, max_y + pad

	x1 = int(_clamp(x1, 0, img_w - 1))
	y1 = int(_clamp(y1, 0, img_h - 1))
	x2 = int(_clamp(x2, 1, img_w))
	y2 = int(_clamp(y2, 1, img_h))
	if x2 <= x1:
		x2 = min(img_w, x1 + 1)
	if y2 <= y1:
		y2 = min(img_h, y1 + 1)

	cx = ((x1 + x2) / 2.0) / img_w
	cy = ((y1 + y2) / 2.0) / img_h
	bw = (x2 - x1) / img_w
	bh = (y2 - y1) / img_h

	parts: list[str] = [
		str(int(ann.class_id)),
		f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}",
	]

	for (x, y, v) in ann.keypoints:
		parts.append(f"{x / img_w:.6f}")
		parts.append(f"{y / img_h:.6f}")
		parts.append(str(int(v)))

	return " ".join(parts)


def _draw_overlay(img_bgr, ann: Annotation, idx: int, total: int, image_path: Path,
				  num_keypoints: int, bbox_mode: str, label_exists: bool) -> None:
	h, w = img_bgr.shape[:2]

	# bbox
	if ann.bbox is not None:
		x1, y1, x2, y2 = ann.bbox
		cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

	# keypoints
	for i, (x, y, v) in enumerate(ann.keypoints):
		color = (0, 255, 0) if v == 2 else (0, 255, 255) if v == 1 else (0, 0, 255)
		cv2.circle(img_bgr, (x, y), 4, color, -1)
		cv2.putText(img_bgr, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

	# status
	status1 = f"[{idx + 1}/{total}] class={ann.class_id}  kpts={len(ann.keypoints)}/{num_keypoints}  bbox={bbox_mode}"
	status2 = (
		f"{image_path.name}  label={'yes' if label_exists else 'no'}  "
		f"keys: 0/1/2 class | LMB click add-kpt / drag bbox | RMB reset | u undo-kpt | s/Enter save | n next | p prev | q quit"
	)
	cv2.rectangle(img_bgr, (0, 0), (w, 55), (0, 0, 0), -1)
	cv2.putText(img_bgr, status1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
	cv2.putText(img_bgr, status2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main() -> int:
	parser = argparse.ArgumentParser(description="Manual label tool for YOLO keypoint/pose datasets")
	parser.add_argument(
		"--dataset",
		type=str,
		default=str(Path(__file__).resolve().parent / "sc_tip_dataset"),
		help="Dataset folder containing images/ and labels/",
	)
	parser.add_argument(
		"--images",
		type=str,
		default=None,
		help="Override images folder (default: <dataset>/images)",
	)
	parser.add_argument(
		"--labels",
		type=str,
		default=None,
		help="Override labels folder (default: <dataset>/labels)",
	)
	parser.add_argument("--kpts", type=int, default=1, help="Number of keypoints per object")
	parser.add_argument(
		"--bbox-mode",
		choices=["auto", "draw"],
		default="auto",
		help="auto: bbox is generated from keypoints; draw: drag a bbox with the mouse",
	)
	parser.add_argument(
		"--auto-box-size",
		type=int,
		default=40,
		help="Auto bbox size (px) for single-keypoint objects (also used as padding)",
	)
	parser.add_argument(
		"--start",
		type=int,
		default=0,
		help="Start index into sorted image list",
	)
	args = parser.parse_args()

	if args.kpts <= 0:
		raise SystemExit("--kpts must be >= 1")

	dataset_dir = Path(args.dataset).expanduser().resolve()
	images_dir = Path(args.images).expanduser().resolve() if args.images else (dataset_dir / "images")
	labels_dir = Path(args.labels).expanduser().resolve() if args.labels else (dataset_dir / "labels")
	labels_dir.mkdir(parents=True, exist_ok=True)

	if not images_dir.exists():
		raise SystemExit(f"Images directory does not exist: {images_dir}")

	image_files = _sorted_image_files(images_dir)
	if not image_files:
		raise SystemExit(f"No images found in: {images_dir}")

	idx = int(_clamp(args.start, 0, len(image_files) - 1))
	window = "yolo-keypoint-labeler"

	# mouse state
	drag_start: Optional[tuple[int, int]] = None
	dragging_bbox = False
	preview_bbox: Optional[tuple[int, int, int, int]] = None
	drag_moved = False

	ann = Annotation(class_id=0)

	def load_annotation_for_current(image_path: Path, img_w: int, img_h: int) -> Annotation:
		label_file = _label_path(labels_dir, image_path)
		existing = _parse_existing_label(label_file, args.kpts)
		if existing is None:
			return Annotation(class_id=0)
		return _hydrate_from_normalized(existing, img_w, img_h, args.kpts)

	def normalize_bbox(b: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
		x1, y1, x2, y2 = b
		x1 = int(_clamp(x1, 0, img_w - 1))
		y1 = int(_clamp(y1, 0, img_h - 1))
		x2 = int(_clamp(x2, 0, img_w - 1))
		y2 = int(_clamp(y2, 0, img_h - 1))
		if x2 < x1:
			x1, x2 = x2, x1
		if y2 < y1:
			y1, y2 = y2, y1
		if x2 == x1:
			x2 = min(img_w - 1, x1 + 1)
		if y2 == y1:
			y2 = min(img_h - 1, y1 + 1)
		return x1, y1, x2, y2

	current_img = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
	if current_img is None:
		raise SystemExit(f"Failed to read image: {image_files[idx]}")
	h0, w0 = current_img.shape[:2]
	ann = load_annotation_for_current(image_files[idx], w0, h0)

	def on_mouse(event, x, y, flags, param) -> None:
		nonlocal drag_start, dragging_bbox, preview_bbox, ann, drag_moved
		if event == cv2.EVENT_RBUTTONDOWN:
			# reset annotation
			ann.bbox = None
			ann.keypoints = []
			drag_start = None
			dragging_bbox = False
			preview_bbox = None
			drag_moved = False
			return

		if args.bbox_mode == "draw":
			if event == cv2.EVENT_LBUTTONDOWN:
				drag_start = (x, y)
				dragging_bbox = True
				preview_bbox = None
				drag_moved = False
			elif event == cv2.EVENT_MOUSEMOVE and dragging_bbox and drag_start is not None:
				x1, y1 = drag_start
				preview_bbox = (x1, y1, x, y)
				if abs(x - x1) + abs(y - y1) >= 6:
					drag_moved = True
			elif event == cv2.EVENT_LBUTTONUP and dragging_bbox and drag_start is not None:
				dragging_bbox = False
				x1, y1 = drag_start
				preview_bbox = None
				if drag_moved:
					# drag => set bbox
					img_h, img_w = current_img.shape[:2]
					ann.bbox = normalize_bbox((x1, y1, x, y), img_w, img_h)
				else:
					# click => add a keypoint
					if len(ann.keypoints) < args.kpts:
						ann.keypoints.append((x, y, 2))
				drag_moved = False
		else:
			# auto bbox mode: click to add keypoints
			if event == cv2.EVENT_LBUTTONDOWN:
				if len(ann.keypoints) >= args.kpts:
					return
				ann.keypoints.append((x, y, 2))

	cv2.namedWindow(window, cv2.WINDOW_NORMAL)
	cv2.setMouseCallback(window, on_mouse)

	while True:
		image_path = image_files[idx]
		img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
		if img is None:
			print(f"[WARN] Failed to read {image_path}, skipping")
			idx = min(len(image_files) - 1, idx + 1)
			continue

		img_h, img_w = img.shape[:2]
		label_file = _label_path(labels_dir, image_path)
		label_exists = label_file.exists() and label_file.stat().st_size > 0

		# Draw overlay on a copy.
		vis = img.copy()
		if preview_bbox is not None and drag_start is not None:
			x1, y1, x2, y2 = preview_bbox
			b = normalize_bbox((x1, y1, x2, y2), img_w, img_h)
			cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 2)

		_draw_overlay(vis, ann, idx, len(image_files), image_path, args.kpts, args.bbox_mode, label_exists)
		cv2.imshow(window, vis)

		key = cv2.waitKey(20) & 0xFF
		if key == 255:
			continue

		if key in (ord('q'), 27):  # q or ESC
			break

		if key in (ord('0'), ord('1'), ord('2')):
			ann.class_id = int(chr(key))
			continue

		# Next / previous image
		if key in (ord('n'), ord('d')):
			idx = min(len(image_files) - 1, idx + 1)
			current_img = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
			if current_img is not None:
				hh, ww = current_img.shape[:2]
				ann = load_annotation_for_current(image_files[idx], ww, hh)
			drag_start = None
			dragging_bbox = False
			preview_bbox = None
			drag_moved = False
			continue

		if key in (ord('p'), ord('a')):
			idx = max(0, idx - 1)
			current_img = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
			if current_img is not None:
				hh, ww = current_img.shape[:2]
				ann = load_annotation_for_current(image_files[idx], ww, hh)
			drag_start = None
			dragging_bbox = False
			preview_bbox = None
			drag_moved = False
			continue

		# Remove last keypoint
		if key in (ord('u'),):
			if ann.keypoints:
				ann.keypoints.pop()
			continue

		# Save
		if key in (ord('s'), 13):  # s or Enter
			try:
				line = _ann_to_yolo_line(
					ann,
					img_w=img_w,
					img_h=img_h,
					num_keypoints=args.kpts,
					bbox_mode=args.bbox_mode,
					auto_box_size_px=args.auto_box_size,
				)
			except Exception as e:
				print(f"[NOT SAVED] {image_path.name}: {e}")
				continue

			label_file.write_text(line + "\n", encoding="utf-8")
			print(f"[SAVED] {label_file}")

			# Auto-advance
			if idx < len(image_files) - 1:
				idx += 1
				current_img = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
				if current_img is not None:
					hh, ww = current_img.shape[:2]
					ann = load_annotation_for_current(image_files[idx], ww, hh)
				drag_start = None
				dragging_bbox = False
				preview_bbox = None
				drag_moved = False
			continue

	cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

