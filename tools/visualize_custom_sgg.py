import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize custom SGG JSON output")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--info-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-thresh", default=0.5, type=float)
    parser.add_argument("--rel-thresh", default=0.5, type=float)
    parser.add_argument("--top-rels", default=20, type=int)
    return parser.parse_args()


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prediction_json, "r") as f:
        predictions = json.load(f)
    with open(args.info_json, "r") as f:
        info = json.load(f)

    classes = info["ind_to_classes"]
    predicates = info["ind_to_predicates"]
    readable = []
    font = ImageFont.load_default()

    for idx in sorted(predictions.keys(), key=lambda x: int(x)):
        pred = predictions[idx]
        image_path = info["idx_to_files"][int(idx)]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        keep = [
            i for i, score in enumerate(pred["bbox_scores"])
            if score >= args.box_thresh
        ]
        kept = set(keep)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep)}

        objects = []
        for old_idx in keep:
            box = pred["bbox"][old_idx]
            label_id = int(pred["bbox_labels"][old_idx])
            score = float(pred["bbox_scores"][old_idx])
            label = classes[label_id]
            objects.append({
                "object_id": index_map[old_idx],
                "source_object_id": old_idx,
                "bbox": box,
                "label_id": label_id,
                "label": label,
                "score": score,
            })
            draw.rectangle(box, outline=(230, 40, 40), width=3)
            draw.text((box[0], max(0, box[1] - 12)), "{} {:.2f}".format(label, score), fill=(230, 40, 40), font=font)

        rel_candidates = []
        for pair, label_id, score in zip(pred["rel_pairs"], pred["rel_labels"], pred["rel_scores"]):
            subj_idx, obj_idx = pair
            score = float(score)
            if subj_idx not in kept or obj_idx not in kept or score < args.rel_thresh:
                continue
            rel_candidates.append((score, subj_idx, obj_idx, int(label_id)))
        rel_candidates.sort(reverse=True)
        rel_candidates = rel_candidates[:args.top_rels]

        relations = []
        for score, subj_idx, obj_idx, label_id in rel_candidates:
            subj_box = pred["bbox"][subj_idx]
            obj_box = pred["bbox"][obj_idx]
            sx, sy = center(subj_box)
            ox, oy = center(obj_box)
            predicate = predicates[label_id]
            draw.line((sx, sy, ox, oy), fill=(40, 120, 255), width=2)
            tx = (sx + ox) / 2.0
            ty = (sy + oy) / 2.0
            draw.text((tx, ty), "{} {:.2f}".format(predicate, score), fill=(40, 120, 255), font=font)
            relations.append({
                "subject_id": index_map[subj_idx],
                "object_id": index_map[obj_idx],
                "source_subject_id": subj_idx,
                "source_object_id": obj_idx,
                "predicate_id": label_id,
                "predicate": predicate,
                "score": score,
            })

        output_name = os.path.splitext(os.path.basename(image_path))[0] + "_sgg.jpg"
        image.save(os.path.join(args.output_dir, output_name), quality=95)
        readable.append({
            "image_index": int(idx),
            "image_path": os.path.abspath(image_path),
            "objects": objects,
            "relations": relations,
            "visualization": os.path.abspath(os.path.join(args.output_dir, output_name)),
        })

    with open(os.path.join(args.output_dir, "readable_predictions.json"), "w") as f:
        json.dump(readable, f, indent=2)

    print("Saved visualizations to {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
