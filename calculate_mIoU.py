"""Compute mIoU between SAM segmentation outputs and ground-truth masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute average mIoU for binary masks.",
	)
	parser.add_argument(
		"--pred-dir",
		type=str,
		default="sam_seg_results/datasets/images/val",
		help="Directory containing predicted segmentation masks.",
	)
	parser.add_argument(
		"--gt-dir",
		type=str,
		default="sam_seg_gt/datasets/images/val",
		help="Directory containing ground-truth segmentation masks.",
	)
	parser.add_argument(
		"--suffix",
		type=str,
		default=".png",
		help="Mask filename suffix to evaluate.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-image IoU.",
	)
	return parser.parse_args()


def _load_binary_mask(mask_path: Path) -> np.ndarray:
	mask = np.array(Image.open(mask_path).convert("L"))
	return mask > 0


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
	intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64)
	union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64)
	if union == 0:
		return 1.0
	return float(intersection / union)


def main() -> None:
	args = parse_args()
	pred_dir = Path(args.pred_dir).expanduser().resolve()
	gt_dir = Path(args.gt_dir).expanduser().resolve()

	if not pred_dir.is_dir():
		raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
	if not gt_dir.is_dir():
		raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

	pred_files = sorted(p for p in pred_dir.rglob(f"*{args.suffix}") if p.is_file())
	if not pred_files:
		raise RuntimeError(f"No prediction files with suffix '{args.suffix}' found in {pred_dir}")

	ious: list[float] = []
	missing_gt: list[Path] = []
	shape_mismatch: list[tuple[Path, tuple[int, int], tuple[int, int]]] = []

	for pred_path in pred_files:
		rel_path = pred_path.relative_to(pred_dir)
		gt_path = gt_dir / rel_path

		if not gt_path.exists():
			missing_gt.append(rel_path)
			continue

		pred_mask = _load_binary_mask(pred_path)
		gt_mask = _load_binary_mask(gt_path)

		if pred_mask.shape != gt_mask.shape:
			shape_mismatch.append((rel_path, pred_mask.shape, gt_mask.shape))
			continue

		iou = _compute_iou(pred_mask, gt_mask)
		ious.append(iou)

		if args.verbose:
			print(f"{rel_path}: IoU={iou:.6f}")

	if missing_gt:
		print(f"[Warning] {len(missing_gt)} prediction files have no matching GT and were skipped.")
	if shape_mismatch:
		print(f"[Warning] {len(shape_mismatch)} files have shape mismatch and were skipped.")

	if not ious:
		raise RuntimeError("No valid prediction/GT pairs were evaluated. Please check paths and files.")

	miou = float(np.mean(ious))
	# print(f"Evaluated pairs: {len(ious)}")
	print(f"Average mIoU: {miou:.6f}")


if __name__ == "__main__":
	main()
