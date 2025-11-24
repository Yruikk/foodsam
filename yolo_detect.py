"""Run YOLOv5 detection on images and export bounding boxes to npy files."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from yolo_utils.plots import Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolo_models.common import DetectMultiBackend
from yolo_utils.dataloaders import LoadImages
from yolo_utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from yolo_utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    image_dir,
    weights=ROOT / "yolo_ckpts" / "yolov5s.pt",
    data=ROOT / "data/mydata2.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    classes=None,
    agnostic_nms=False,
    half=False,
    dnn=False,
    output_dir=ROOT / "yolo_seg_results",
    augment=False,
    visualize=False,
):
    source_dir = Path(image_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {source_dir}")

    results_dir = Path(output_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if visualize:
        visual_dir = ROOT / "yolo_seg_results_visual"
        visual_dir.mkdir(parents=True, exist_ok=True)

    dataset_anchor = None
    for candidate in [source_dir, *source_dir.parents]:
        if candidate.name == "datasets":
            dataset_anchor = candidate
            break
    if dataset_anchor is not None:
        source_subdir = source_dir.relative_to(dataset_anchor)
    else:
        source_subdir = Path(source_dir.name)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, _, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(str(source_dir), img_size=imgsz, stride=stride, auto=pt)
    profiler = (Profile(), Profile(), Profile())
    processed = 0

    for path, im, im0s, _, _ in dataset:
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".jpg":
            continue

        with profiler[0]:
            tensor = torch.from_numpy(im).to(model.device)
            tensor = tensor.half() if model.fp16 else tensor.float()
            tensor /= 255
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        with profiler[1]:
            predictions = model(tensor, augment=augment)

        with profiler[2]:
            detections = non_max_suppression(
                predictions,
                conf_thres,
                iou_thres,
                classes,
                agnostic_nms,
                max_det=max_det,
            )

        for det in detections:
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], im0.shape).round()
                output = det[:, :4].cpu().numpy().astype(np.float32)
            else:
                output = np.empty((0, 4), dtype=np.float32)

            try:
                rel_path = path_obj.resolve().relative_to(source_dir)
            except ValueError:
                rel_path = Path(path_obj.name)
            target_path = (results_dir / source_subdir / rel_path).with_suffix(".npy")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, output)
            processed += 1
            LOGGER.info(f"{colorstr('green', 'Saved')} {output.shape[0]} boxes -> {target_path}")

            # Visualization
            if visualize:
                visual_target_path = (visual_dir / source_subdir / rel_path)
                visual_target_path.parent.mkdir(parents=True, exist_ok=True)

                annotator = Annotator(im0, line_width=3, example=str(model.names))
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

                cv2.imwrite(str(visual_target_path), annotator.result())

    LOGGER.info(f"{colorstr('blue', 'Done')} Processed {processed} images from {source_dir}")


def parse_opt():
    parser = argparse.ArgumentParser(description="Run YOLO detection on JPG folders and export npy results.")
    parser.add_argument("--image-dir", type=str, required=True, help="Folder containing input JPG images.")
    parser.add_argument("--food-class", type=str, default=None, help="Food class keyword to select YOLO weights.")
    parser.add_argument("--data", type=str, default=ROOT / "data/mydata2.yaml", help="Dataset yaml path.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Inference size h,w.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image.")
    parser.add_argument("--device", type=str, default="", help="CUDA device or cpu.")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter by class ids.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS.")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision inference.")
    parser.add_argument("--dnn", action="store_true", help="Use OpenCV DNN for ONNX inference.")
    parser.add_argument("--output-dir", type=str, default=ROOT / "yolo_seg_results", help="Directory for npy files.")
    parser.add_argument("--augment", action="store_true", help="Run inference with augmentation.")
    parser.add_argument("--yolo-visualize", action="store_true", help="Visualize YOLO detection results.")
    return parser.parse_args()


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    imgsz = opt.imgsz if len(opt.imgsz) > 1 else opt.imgsz * 2
    imgsz = tuple(imgsz)
    weights_root = ROOT / "yolo_ckpts"
    if opt.food_class:
        weight_path = weights_root / f"{opt.food_class}.pt"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Weight file not found for food class '{opt.food_class}': {weight_path}")
    else:
        weight_path = weights_root / "yolov5s_v1.pt"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Default weight file missing: {weight_path}")

    arguments = {
        "image_dir": opt.image_dir,
        "weights": weight_path,
        "data": opt.data,
        "imgsz": imgsz,
        "conf_thres": opt.conf_thres,
        "iou_thres": opt.iou_thres,
        "max_det": opt.max_det,
        "device": opt.device,
        "classes": opt.classes,
        "agnostic_nms": opt.agnostic_nms,
        "half": opt.half,
        "dnn": opt.dnn,
        "output_dir": opt.output_dir,
        "augment": opt.augment,
        "visualize": opt.yolo_visualize,
    }
    print_args({**arguments, "food_class": opt.food_class, "weights": str(weight_path)})
    run(**arguments)


if __name__ == "__main__":
    cli_opt = parse_opt()
    main(cli_opt)
