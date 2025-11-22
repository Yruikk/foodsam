"""Pipeline entry point: run YOLO detection, then SAM segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import sam_seg
import yolo_detect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform YOLO detection to generate box prompts, then run SAM segmentation.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing the input JPG images (processed recursively).",
    )
    parser.add_argument(
        "--food-class",
        type=str,
        default=None,
        help="Food class keyword used for both YOLO weights and SAM weight lookup.",
    )

    det_group = parser.add_argument_group("YOLO detection options")
    det_group.add_argument(
        "--det-weights",
        type=str,
        default=None,
        help="Explicit path to a YOLO weight file. Overrides --food-class selection.",
    )
    det_group.add_argument(
        "--det-data",
        type=str,
        default=str(yolo_detect.ROOT / "data/mydata2.yaml"),
        help="Path to the YOLO dataset YAML file.",
    )
    det_group.add_argument(
        "--det-imgsz",
        nargs="+",
        type=int,
        default=[640],
        help="YOLO inference image size (height width). Provide one value for square size.",
    )
    det_group.add_argument(
        "--det-conf-thres",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    det_group.add_argument(
        "--det-iou-thres",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold.",
    )
    det_group.add_argument(
        "--det-max-det",
        type=int,
        default=1000,
        help="Maximum number of detections per image.",
    )
    det_group.add_argument(
        "--det-device",
        type=str,
        default="",
        help="YOLO inference device identifier (e.g. 'cuda:0' or 'cpu').",
    )
    det_group.add_argument(
        "--det-classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter YOLO detections by class indices.",
    )
    det_group.add_argument(
        "--det-agnostic-nms",
        action="store_true",
        help="Enable class-agnostic YOLO NMS.",
    )
    det_group.add_argument(
        "--det-half",
        action="store_true",
        help="Run YOLO in FP16 mode.",
    )
    det_group.add_argument(
        "--det-dnn",
        action="store_true",
        help="Use OpenCV DNN for ONNX YOLO inference.",
    )
    det_group.add_argument(
        "--det-output-dir",
        type=str,
        default=str(yolo_detect.ROOT / "yolo_seg_results"),
        help="Directory where YOLO bounding boxes (npy) are stored.",
    )
    det_group.add_argument(
        "--det-augment",
        action="store_true",
        help="Run YOLO inference with augmentation.",
    )

    sam_group = parser.add_argument_group("SAM segmentation options")
    sam_group.add_argument(
        "--sam-boxes-root",
        type=str,
        default=None,
        help="Root directory containing YOLO box prompts (defaults to --det-output-dir).",
    )
    sam_group.add_argument(
        "--sam-output",
        type=str,
        default="sam_seg_results",
        help="Directory to store SAM binary segmentation outputs.",
    )
    sam_group.add_argument(
        "--sam-checkpoint",
        type=str,
        default="sam_ckpts/sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file.",
    )
    sam_group.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        help="SAM model type (e.g. 'vit_h', 'vit_l', 'vit_b').",
    )
    sam_group.add_argument(
        "--sam-device",
        type=str,
        default="cuda:0",
        help="SAM inference device identifier.",
    )
    sam_group.add_argument(
        "--sam-food-class",
        type=str,
        default=None,
        help="Override the food class used during SAM weight lookup.",
    )

    return parser.parse_args()


def _prepare_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 0:
        raise ValueError("--det-imgsz requires at least one integer value.")
    if len(values) == 1:
        side = values[0]
        return side, side
    return values[0], values[1]


def run_pipeline(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    yolo_detect.check_requirements(exclude=("tensorboard", "thop"))

    weights_root = yolo_detect.ROOT / "yolo_ckpts"
    if args.det_weights:
        weights_path = Path(args.det_weights)
    elif args.food_class:
        weights_path = weights_root / f"{args.food_class}.pt"
    else:
        weights_path = weights_root / "yolov5s.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weight file not found: {weights_path}")

    imgsz = _prepare_imgsz(args.det_imgsz)

    detection_kwargs = {
        "image_dir": str(image_dir),
        "weights": weights_path,
        "data": Path(args.det_data),
        "imgsz": imgsz,
        "conf_thres": args.det_conf_thres,
        "iou_thres": args.det_iou_thres,
        "max_det": args.det_max_det,
        "device": args.det_device,
        "classes": args.det_classes,
        "agnostic_nms": args.det_agnostic_nms,
        "half": args.det_half,
        "dnn": args.det_dnn,
        "output_dir": Path(args.det_output_dir),
        "augment": args.det_augment,
    }

    yolo_detect.run(**detection_kwargs)

    boxes_root = Path(args.sam_boxes_root) if args.sam_boxes_root else Path(args.det_output_dir)
    sam_args = argparse.Namespace(
        image_dir=str(image_dir),
        boxes_root=str(boxes_root),
        output=args.sam_output,
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
        food_class=args.sam_food_class if args.sam_food_class else args.food_class,
    )

    sam_seg.run(sam_args)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
