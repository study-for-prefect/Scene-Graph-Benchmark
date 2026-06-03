# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Run detector-only inference on a folder of images and export JSON + previews."""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Custom detector inference")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--score-thresh", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_label_names(output_dir):
    label_file = os.path.join(output_dir, "labels.json")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            return json.load(f)
    return {}


def get_field(boxlist, *names):
    for name in names:
        if boxlist.has_field(name):
            return boxlist.get_field(name)
    raise KeyError("None of the requested fields exist: {}".format(names))


def list_images(input_dir):
    images = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def draw_preview(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = "{} {:.3f}".format(det["label_name"], det["score"])
        draw.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=3)
        text_y = y1 - 12 if y1 >= 12 else y2 + 2
        draw.text((x1, text_y), label, fill=(230, 40, 40), font=font)
    image.save(output_path, quality=95)


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHT = args.weight
    if device == "cpu":
        cfg.DTYPE = "float32"
    cfg.freeze()

    mkdir(args.output_dir)
    json_dir = os.path.join(args.output_dir, "json")
    vis_dir = os.path.join(args.output_dir, "vis")
    mkdir(json_dir)
    mkdir(vis_dir)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir="")
    checkpointer.load(args.weight, with_optim=False)

    transforms = build_transforms(cfg, is_train=False)
    label_names = load_label_names(cfg.OUTPUT_DIR)
    image_paths = list_images(args.input_dir)
    results = []

    for image_id, image_path in enumerate(image_paths):
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
        for box, label, score in sorted(zip(boxes, labels, scores), key=lambda x: x[2], reverse=True):
            if score < args.score_thresh:
                continue
            label_id = int(label)
            detections.append(
                {
                    "bbox": [float(v) for v in box],
                    "label_id": label_id,
                    "label_name": label_names.get(str(label_id), str(label_id)),
                    "score": float(score),
                }
            )

        stem = os.path.splitext(os.path.basename(image_path))[0]
        image_result = {
            "image_id": image_id,
            "file_name": os.path.basename(image_path),
            "image_path": os.path.abspath(image_path),
            "width": original_size[0],
            "height": original_size[1],
            "score_threshold": args.score_thresh,
            "detections": detections,
        }
        with open(os.path.join(json_dir, "{}.json".format(stem)), "w") as f:
            json.dump(image_result, f, indent=2)
        draw_preview(image_path, detections, os.path.join(vis_dir, "{}.jpg".format(stem)))
        results.append(image_result)

    with open(os.path.join(args.output_dir, "all_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Processed {} images".format(len(results)))
    print("JSON: {}".format(os.path.abspath(json_dir)))
    print("Visualization: {}".format(os.path.abspath(vis_dir)))


if __name__ == "__main__":
    main()
