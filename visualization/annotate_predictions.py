import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (69, 123, 157),
    (131, 56, 236),
    (255, 0, 110),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(project_root, raw_path):
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return (project_root / image_path).resolve()


def get_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_box(draw, xy, text, font, fill, text_fill=(255, 255, 255)):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    pad_x, pad_y = 4, 2
    draw.rectangle(
        (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
        fill=fill,
    )
    draw.text((x, y), text, font=font, fill=text_fill)


def arrow(draw, start, end, fill, width):
    draw.line((start, end), fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = max(8, width * 4)
    spread = math.pi / 7
    p1 = (
        ex - length * math.cos(angle - spread),
        ey - length * math.sin(angle - spread),
    )
    p2 = (
        ex - length * math.cos(angle + spread),
        ey - length * math.sin(angle + spread),
    )
    draw.polygon((end, p1, p2), fill=fill)


def draw_annotation(image, pred, info, args):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = max(12, int(min(width, height) * 0.018))
    small_font_size = max(11, int(font_size * 0.85))
    font = get_font(font_size)
    small_font = get_font(small_font_size)
    line_width = max(2, int(min(width, height) * 0.003))

    classes = info["ind_to_classes"]
    predicates = info["ind_to_predicates"]

    boxes = pred["bbox"][: args.box_topk]
    labels = pred["bbox_labels"][: args.box_topk]
    scores = pred["bbox_scores"][: args.box_topk]

    active = set()
    selected_relations = []
    for pair, predicate_idx, score in zip(
        pred["rel_pairs"], pred["rel_labels"], pred["rel_scores"]
    ):
        if len(selected_relations) >= args.rel_topk:
            break
        if score < args.rel_thresh:
            continue
        subj_idx, obj_idx = pair
        if subj_idx >= len(boxes) or obj_idx >= len(boxes):
            continue
        if scores[subj_idx] < args.box_thresh or scores[obj_idx] < args.box_thresh:
            continue
        selected_relations.append((subj_idx, obj_idx, predicate_idx, score))
        active.add(subj_idx)
        active.add(obj_idx)

    if args.show_all_boxes:
        active.update(range(len(boxes)))

    centers = {}
    for idx, box in enumerate(boxes):
        if idx not in active:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        color = PALETTE[idx % len(PALETTE)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        centers[idx] = ((x1 + x2) / 2, (y1 + y2) / 2)

        class_name = classes[labels[idx]]
        label = f"{idx} {class_name} {scores[idx]:.2f}"
        label_y = max(2, y1 - font_size - 6)
        text_box(draw, (x1 + 3, label_y), label, font, color)

    if args.draw_relations:
        for subj_idx, obj_idx, predicate_idx, score in selected_relations:
            if subj_idx not in centers or obj_idx not in centers:
                continue
            color = PALETTE[subj_idx % len(PALETTE)]
            start = centers[subj_idx]
            end = centers[obj_idx]
            arrow(draw, start, end, color, max(2, line_width - 1))
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            relation = f"{predicates[predicate_idx]} {score:.2f}"
            text_box(draw, (mid[0] + 4, mid[1] + 4), relation, small_font, (0, 0, 0))

    return len(active), len(selected_relations)


def annotate_all(args):
    project_root = Path(args.project_root).resolve()
    info = load_json(project_root / args.info)
    predictions = load_json(project_root / args.predictions)
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_indices = range(len(info["idx_to_files"]))
    if args.image_idx is not None:
        image_indices = [args.image_idx]

    for image_idx in image_indices:
        key = str(image_idx)
        if key not in predictions:
            print(f"skip image {image_idx}: no prediction")
            continue

        image_path = resolve_image_path(project_root, info["idx_to_files"][image_idx])
        if not image_path.exists():
            print(f"skip image {image_idx}: missing image {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        box_count, rel_count = draw_annotation(image, predictions[key], info, args)

        output_path = output_dir / f"{image_idx}_{image_path.stem}_annotated.jpg"
        image.save(output_path, quality=95)
        print(
            f"saved {output_path} "
            f"({box_count} boxes, {rel_count} relations)"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw scene graph prediction annotations on images."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--info", default="output_dir/custom_data_info.json")
    parser.add_argument("--predictions", default="output_dir/custom_prediction.json")
    parser.add_argument("--output-dir", default="output_dir/annotated_images")
    parser.add_argument("--image-idx", type=int, default=None)
    parser.add_argument("--box-topk", type=int, default=50)
    parser.add_argument("--rel-topk", type=int, default=12)
    parser.add_argument("--box-thresh", type=float, default=0.02)
    parser.add_argument("--rel-thresh", type=float, default=0.10)
    parser.add_argument("--show-all-boxes", action="store_true")
    parser.add_argument("--no-relations", dest="draw_relations", action="store_false")
    parser.set_defaults(draw_relations=True)
    return parser.parse_args()


if __name__ == "__main__":
    annotate_all(parse_args())
