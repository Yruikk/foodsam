"""Compute global weight estimation accuracy from saved per-image predictions.

Expected structure under weight_results:
    .../<count>-<groundtruth_weight>/<image_name>.npy

Each .npy stores one predicted weight value (grams).
Metric:
    RSE = |pred - gt| / gt
    global_avg_RSE = mean(RSE over all valid npy files)
    accuracy = 1 - global_avg_RSE
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


LABEL_PATTERN = re.compile(r"^(\d+)-(\d+(?:\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate global average RSE and accuracy from weight_results npy files.",
    )
    parser.add_argument(
        "--weight-results-root",
        type=str,
        default="weight_results",
        help="Root directory containing per-image predicted weight .npy files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file RSE details.",
    )
    return parser.parse_args()


def _find_label_dir(path: Path, root: Path) -> tuple[int, float] | None:
    """Find nearest ancestor dir named <count>-<weight> between file parent and root."""
    current = path.parent
    root = root.resolve()
    while True:
        m = LABEL_PATTERN.match(current.name)
        if m:
            return int(m.group(1)), float(m.group(2))
        if current == root:
            break
        if current.parent == current:
            break
        current = current.parent
    return None


def _read_pred_weight(npy_path: Path) -> float:
    arr = np.load(npy_path)
    if np.size(arr) == 0:
        raise ValueError("empty npy")
    return float(np.ravel(arr)[0])


def main() -> None:
    args = parse_args()
    root = (Path(__file__).resolve().parent / args.weight_results_root).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"weight_results root not found: {root}")

    npy_files = sorted(p for p in root.rglob("*.npy") if p.is_file())
    if not npy_files:
        raise RuntimeError(f"No .npy files found under: {root}")

    rse_values: list[float] = []
    skipped = 0

    for npy_path in npy_files:
        label = _find_label_dir(npy_path, root)
        if label is None:
            skipped += 1
            if args.verbose:
                print(f"[Skip] No <count>-<weight> folder in path: {npy_path}")
            continue

        _, gt_weight = label
        if gt_weight <= 0:
            skipped += 1
            if args.verbose:
                print(f"[Skip] Non-positive gt weight in path: {npy_path}")
            continue

        try:
            pred_weight = _read_pred_weight(npy_path)
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"[Skip] Bad npy ({exc}): {npy_path}")
            continue

        rse = abs(pred_weight - gt_weight) / gt_weight
        rse_values.append(float(rse))

        if args.verbose:
            rel = npy_path.relative_to(root)
            print(f"{rel} | pred={pred_weight:.4f}g | gt={gt_weight:.4f}g | RSE={rse:.6f}")

    if not rse_values:
        raise RuntimeError("No valid prediction files found for RSE calculation.")

    avg_rse = float(np.mean(rse_values))
    accuracy = 1.0 - avg_rse

    print("===== Global Weight Accuracy =====")
    print(f"Weight results root: {root}")
    # print(f"Total npy files: {len(npy_files)}")
    print(f"Valid predictions: {len(rse_values)}")
    # print(f"Skipped files: {skipped}")
    # print(f"Average RSE: {avg_rse:.6f}")
    print(f"Accuracy: {accuracy:.6f}")


if __name__ == "__main__":
    main()
