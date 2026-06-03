# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Quick visual check for the custom detector checkpoint.

Defaults are wired for custom_output_detector/model_final.pth so this can be
run without arguments from the repository root.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import json

import torch
from PIL import Image, ImageDraw, ImageFont

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir


DEFAULT_CONFIG = os.path.join(ROOT_DIR, "custom_output_detector", "config.yml")
DEFAULT_WEIGHT = os.path.join(ROOT_DIR, "custom_output_detector", "model_final.pth")
DEFAULT_INPUT = os.path.join(ROOT_DIR, "input_dir")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "custom_output_detector", "check_model_final")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detector inference with custom_output_detector/model_final.pth"
    )
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, type=str)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, type=str)
    parser.add_argument("--input", default=DEFAULT_INPUT, type=str, help="Image file or image directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, type=str)
    parser.add_argument("--score-thresh", default=0.5, type=float)
    parser.add_argument("--max-detections", default=50, type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "opts",
        help="Optional config overrides, e.g. MODEL.ROI_HEADS.SCORE_THRESH 0.05",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT_DIR, path))


def load_label_names(checkpoint_dir):
    label_file = os.path.join(checkpoint_dir, "labels.json")
    if not os.path.exists(label_file):
        return {}
    with open(label_file, "r") as f:
        labels = json.load(f)
    if isinstance(labels, list):
        return {str(index): name for index, name in enumerate(labels)}
    return labels


def get_field(boxlist, *names):
    for name in names:
        if boxlist.has_field(name):
            return boxlist.get_field(name)
    raise KeyError("None of the requested fields exist: {}".format(names))


def list_images(input_path):
    input_path = resolve_path(input_path)
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path.lower())[1]
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image extension: {}".format(input_path))
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError("Input path does not exist: {}".format(input_path))

    image_paths = []
    for name in sorted(os.listdir(input_path)):
        path = os.path.join(input_path, name)
        ext = os.path.splitext(name.lower())[1]
        if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
            image_paths.append(path)
    if not image_paths:
        raise ValueError("No images found in: {}".format(input_path))
    return image_paths


def color_for_label(label_id):
    value = int(label_id) * 1103515245 + 12345
    return (
        40 + value % 180,
        40 + (value // 97) % 180,
        40 + (value // 193) % 180,
    )


def draw_preview(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        color = color_for_label(det["label_id"])
        label = "{} {:.2f}".format(det["label_name"], det["score"])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w = right - left
            text_h = bottom - top
        else:
            text_w, text_h = draw.textsize(label, font=font)
        text_x = max(0, int(x1))
        text_y = int(y1) - text_h - 4
        if text_y < 0:
            text_y = min(int(y2) + 2, image.height - text_h - 2)
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4],
            fill=color,
        )
        draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

    image.save(output_path, quality=95)


def build_model(args):
    config_file = resolve_path(args.config_file)
    weight = resolve_path(args.weight)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = "cpu"
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHT = weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH = min(cfg.MODEL.ROI_HEADS.SCORE_THRESH, args.score_thresh)
    if device == "cpu":
        cfg.DTYPE = "float32"
    cfg.freeze()

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir="")
    checkpointer.load(weight, with_optim=False)
    return model


def run_one_image(model, transforms, image_path, label_names, args):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    empty_target = BoxList(torch.empty((0, 4)), image.size, mode="xyxy")
    tensor, _ = transforms(image, empty_target)

    with torch.no_grad():
        image_list = to_image_list([tensor], cfg.DATALOADER.SIZE_DIVISIBILITY).to(cfg.MODEL.DEVICE)
        prediction = model(image_list)[0].to("cpu")

    prediction = prediction.resize(original_size).convert("xyxy")
    labels = get_field(prediction, "pred_labels", "labels").tolist()
    scores = get_field(prediction, "pred_scores", "scores").tolist()
    boxes = prediction.bbox.tolist()

    detections = []
    ranked = sorted(zip(boxes, labels, scores), key=lambda item: item[2], reverse=True)
    for box, label, score in ranked:
        if score < args.score_thresh:
            continue
        label_id = int(label)
        detections.append(
            {
                "bbox": [round(float(v), 2) for v in box],
                "label_id": label_id,
                "label_name": label_names.get(str(label_id), str(label_id)),
                "score": round(float(score), 6),
            }
        )
        if len(detections) >= args.max_detections:
            break

    return {
        "file_name": os.path.basename(image_path),
        "image_path": os.path.abspath(image_path),
        "width": original_size[0],
        "height": original_size[1],
        "num_detections": len(detections),
        "detections": detections,
    }


def main():
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.weight = resolve_path(args.weight)

    mkdir(args.output_dir)
    json_dir = os.path.join(args.output_dir, "json")
    vis_dir = os.path.join(args.output_dir, "vis")
    mkdir(json_dir)
    mkdir(vis_dir)

    model = build_model(args)
    transforms = build_transforms(cfg, is_train=False)
    label_names = load_label_names(os.path.dirname(args.weight))
    image_paths = list_images(args.input)

    results = []
    for index, image_path in enumerate(image_paths, 1):
        result = run_one_image(model, transforms, image_path, label_names, args)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(json_dir, "{}.json".format(stem)), "w") as f:
            json.dump(result, f, indent=2)
        draw_preview(image_path, result["detections"], os.path.join(vis_dir, "{}.jpg".format(stem)))
        results.append(result)
        print(
            "[{}/{}] {}: {} detections".format(
                index, len(image_paths), result["file_name"], result["num_detections"]
            )
        )

    summary = {
        "config_file": os.path.abspath(resolve_path(args.config_file)),
        "weight": os.path.abspath(args.weight),
        "input": os.path.abspath(resolve_path(args.input)),
        "score_threshold": args.score_thresh,
        "max_detections": args.max_detections,
        "num_images": len(results),
        "results": results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Visualization: {}".format(vis_dir))
    print("Per-image JSON: {}".format(json_dir))
    print("Summary JSON: {}".format(os.path.join(args.output_dir, "summary.json")))


if __name__ == "__main__":
    main()
