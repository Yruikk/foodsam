"""Segment images with SAM using YOLO-derived box prompts."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM segmentation on JPG images using stored box prompts."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input JPG images (recursively processed).",
    )
    parser.add_argument(
        "--boxes-root",
        type=str,
        default="yolo_seg_results",
        help="Root directory where YOLO npy box prompts are stored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sam_seg_results",
        help="Directory to store SAM binary segmentation outputs.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="sam_ckpts/sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        help="SAM model type: one of ['default', 'vit_h', 'vit_l', 'vit_b'].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device identifier, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--food-class",
        type=str,
        default=None,
        help="Food category key used to look up per-pixel weight.",
    )
    parser.add_argument(
        "--sam-visualize",
        action="store_true",
        help="Visualize SAM segmentation results.",
    )
    return parser.parse_args()


def _create_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"sam_seg_{timestamp}.log"

    logger = logging.getLogger("sam_seg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.info("Logger initialized at %s", log_path)
    return logger


def _find_dataset_anchor(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if candidate.name == "datasets":
            return candidate
    return None


def _collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.jpg") if p.is_file())


def _load_weight_config(config_path: Path, logger: logging.Logger) -> dict:
    if not config_path.is_file():
        logger.warning(
            "Weight config %s not found. Falling back to default weight 0.2 g/pixel.",
            config_path,
        )
        return {"default_weight": 0.2, "classes": {}}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to parse %s (%s). Using default weight 0.2 g/pixel.",
            config_path,
            exc,
        )
        return {"default_weight": 0.2, "classes": {}}

    classes = {}
    for key, value in data.get("classes", {}).items():
        try:
            classes[str(key).strip().lower()] = float(value)
        except (TypeError, ValueError):
            logger.warning("Invalid weight for key '%s' in %s", key, config_path)

    default_weight = data.get("default_weight", 0.2)
    try:
        default_weight = float(default_weight)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid default_weight in %s. Using 0.2 g/pixel as fallback.",
            config_path,
        )
        default_weight = 0.2

    return {"default_weight": default_weight, "classes": classes}


def _resolve_weight(food_class: str | None, source_subdir: Path, config: dict) -> tuple[float, str]:
    classes = config.get("classes", {})
    fallback = config.get("default_weight", 0.2)

    def normalize(value: str | Path) -> str:
        return str(value).strip().lower().replace("\\", "/")

    candidates: list[str] = []
    if food_class:
        candidates.append(food_class)

    parts = [part for part in Path(str(source_subdir)).parts if part not in (".",)]
    for i in range(len(parts), 0, -1):
        candidates.append("/".join(parts[:i]))

    seen: set[str] = set()
    for candidate in candidates:
        key = normalize(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in classes:
            return classes[key], f"classes['{key}']"

    return float(fallback), "default_weight"


def run(args: argparse.Namespace) -> None:
    image_root = Path(args.image_dir).expanduser().resolve()
    boxes_root = Path(args.boxes_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not boxes_root.exists():
        raise FileNotFoundError(f"Boxes directory not found: {boxes_root}")
    if not Path(args.sam_checkpoint).expanduser().resolve().exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    if args.sam_visualize:
        visual_root = ROOT / "sam_seg_results_visual"
        visual_root.mkdir(parents=True, exist_ok=True)

    logger = _create_logger(output_root)
    logger.info("Processing images under %s", image_root)

    images = _collect_images(image_root)
    if not images:
        logger.warning("No JPG files discovered. Exiting.")
        return

    dataset_anchor = _find_dataset_anchor(image_root)
    source_subdir = (
        image_root.relative_to(dataset_anchor)
        if dataset_anchor is not None
        else Path(image_root.name)
    )

    config_path = Path(__file__).resolve().parent / "weight_per_pixel.json"
    weight_config = _load_weight_config(config_path, logger)
    weight_per_pixel, weight_source = _resolve_weight(args.food_class, source_subdir, weight_config)
    if args.food_class is None:
        logger.warning(
            "--food-class not provided; using fallback weight %.6f from %s",
            weight_per_pixel,
            weight_source,
        )
    else:
        logger.info(
            "Using per-pixel weight %.6f for '%s' (source: %s)",
            weight_per_pixel,
            args.food_class,
            weight_source,
        )

    logger.info("Using SAM checkpoint %s", args.sam_checkpoint)
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    processed = 0
    skipped = 0
    total_pixels = 0
    total_weight = 0.0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Unable to load image: %s", image_path)
            skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        rel_path = image_path.relative_to(image_root)
        box_path = (boxes_root / source_subdir / rel_path).with_suffix(".npy")
        if not box_path.exists():
            logger.warning("Missing box prompts: %s", box_path)
            skipped += 1
            continue

        boxes = np.load(box_path)
        if boxes.size == 0:
            logger.warning("Empty box prompts: %s", box_path)
            skipped += 1
            continue

        masks_accum = []
        for box in np.atleast_2d(boxes):
            box = box.reshape(1, 4)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            order = np.argsort(scores)[::-1]
            masks_accum.append(masks[order])

        combined = np.any([np.any(m, axis=0) for m in masks_accum], axis=0) if masks_accum else np.zeros(image.shape[:2], dtype=bool)
        binary = np.zeros(combined.shape, dtype=np.uint8)
        binary[combined] = 255

        output_path = (output_root / source_subdir / rel_path).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(output_path)

        if args.sam_visualize:
            visual_path = (visual_root / source_subdir / rel_path).with_suffix(".jpg")
            visual_path.parent.mkdir(parents=True, exist_ok=True)

            overlay = image.copy()
            overlay[binary == 255] = [0, 255, 0]
            vis_image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(visual_path), vis_image)
            if not success:
                logger.error("Failed to save visualization to %s", visual_path)

        white_pixels = int(np.count_nonzero(binary))
        estimated_weight = white_pixels * weight_per_pixel
        total_pixels += white_pixels
        total_weight += estimated_weight
        logger.info(
            "Saved segmentation -> %s | pixels=%d | estimated_weight=%.2fg",
            output_path,
            white_pixels,
            estimated_weight,
        )
        processed += 1


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
