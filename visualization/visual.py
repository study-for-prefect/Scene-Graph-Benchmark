import os
import json
import math
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 基础路径配置 (绝对路径)
INFO_PATH = "/home/wxm/code/Scene-Graph-Benchmark/output_dir/custom_data_info.json"
PRED_PATH = "/home/wxm/code/Scene-Graph-Benchmark/output_dir/custom_prediction.json"

# 图数据库连接配置 (按实际部署环境修改)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 截断超参数
BOX_TOPK = 30
REL_TOPK = 20
BOX_THRESH = 0.02
REL_THRESH = 0.03
VIS_OUTPUT_DIR = "/home/wxm/code/Scene-Graph-Benchmark/output_dir/visualized"
PALETTE = [
    "red", "deepskyblue", "lime", "yellow", "magenta",
    "orange", "cyan", "white", "dodgerblue", "hotpink",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offline or realtime custom scene graphs.")
    parser.add_argument("--mode", choices=("offline", "realtime", "llm"), default="offline")
    parser.add_argument("--info-path", default=INFO_PATH)
    parser.add_argument("--pred-path", default=PRED_PATH)
    parser.add_argument("--scene-graph-json", default="/tmp/realtime_scene_graph_latest.json")
    parser.add_argument("--image-path", default="", help="Raw frame image for realtime visualization.")
    parser.add_argument("--output-dir", default=VIS_OUTPUT_DIR)
    parser.add_argument("--box-topk", type=int, default=BOX_TOPK)
    parser.add_argument("--rel-topk", type=int, default=REL_TOPK)
    parser.add_argument("--box-thresh", type=float, default=BOX_THRESH)
    parser.add_argument("--rel-thresh", type=float, default=REL_THRESH)
    parser.add_argument("--sync-neo4j", action="store_true", help="Sync realtime graph to Neo4j.")
    return parser.parse_args()


class Neo4jManager:
    """图数据库连接池与写入管理器"""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password)) if GraphDatabase else None

    def close(self):
        if self.driver:
            self.driver.close()

    def sync_scene_graph(self, image_idx, valid_relations, box_labels, box_scores):
        """将单帧场景图写入持久化图数据库"""
        if self.driver is None:
            return
        with self.driver.session() as session:
            for rel in valid_relations:
                session.execute_write(self._merge_relation, image_idx, rel, box_labels, box_scores)

    @staticmethod
    def _merge_relation(tx, image_idx, rel, box_labels, box_scores):
        # 构建图谱全局唯一节点ID (基于当前图像索引与实体索引)
        subj_global_id = f"IMG{image_idx}_OBJ{rel['subj_idx']}"
        obj_global_id = f"IMG{image_idx}_OBJ{rel['obj_idx']}"

        # 格式化 Cypher 关系名称 (大写并替换空格)
        rel_type = rel['predicate'].replace(' ', '_').upper()

        # Cypher 写入语句: MERGE 保证节点唯一性，SET 实施属性更新
        query = f"""
        MERGE (s:Entity {{id: $subj_id}})
        SET s.class = $subj_class, s.local_idx = $subj_idx, s.score = $subj_score, s.last_seen_img = $image_idx

        MERGE (o:Entity {{id: $obj_id}})
        SET o.class = $obj_class, o.local_idx = $obj_idx, o.score = $obj_score, o.last_seen_img = $image_idx

        MERGE (s)-[r:`{rel_type}`]->(o)
        SET r.raw_predicate = $predicate, r.score = $rel_score, r.last_seen_img = $image_idx
        """

        tx.run(query,
               subj_id=subj_global_id, subj_class=box_labels[rel['subj_idx']],
               subj_idx=rel['subj_idx'], subj_score=box_scores[rel['subj_idx']],
               obj_id=obj_global_id, obj_class=box_labels[rel['obj_idx']],
               obj_idx=rel['obj_idx'], obj_score=box_scores[rel['obj_idx']],
               predicate=rel['predicate'], rel_score=rel.get('score', 1.0),
               image_idx=image_idx)


def get_size(image_size):
    min_size, max_size = 600, 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_orig_min = float(min((w, h)))
        min_orig_max = float(max((w, h)))
        if min_orig_max / min_orig_min * size > max_size:
            size = int(round(max_size * min_orig_min / min_orig_max))
    if w < h:
        return (size, int(size * h / w))
    return (int(size * w / h), size)


def get_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_text_label(draw, xy, text, font, fill):
    x, y = xy
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    else:
        text_w, text_h = draw.textsize(text, font=font)
        left, top, right, bottom = x, y, x + text_w, y + text_h
    draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill="black")
    draw.text((x, y), text, fill=fill, font=font)


def draw_arrow(draw, start, end, fill, width):
    draw.line((start, end), fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = max(10, width * 5)
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


def object_node_name(obj):
    obj_id = obj.get("object_id", obj.get("id"))
    return "{}_{}".format(obj_id, obj.get("label_name", obj.get("label", obj.get("label_id", "object"))))


def object_graph_label(obj):
    obj_id = obj.get("object_id", obj.get("id"))
    label = "{}:{}".format(obj_id, obj.get("label_name", obj.get("label", obj.get("label_id", "object"))))
    score = obj.get("score", obj.get("confidence"))
    if score is not None:
        label += "\n{:.2f}".format(score)
    areas = obj.get("areas")
    if areas:
        label += "\n{}".format(",".join(areas))
    xyz = obj.get("center_3d_m")
    if xyz:
        label += "\nxyz {:.2f},{:.2f},{:.2f}m".format(xyz[0], xyz[1], xyz[2])
    return label


def filter_realtime_scene_graph(payload, box_topk, rel_topk, box_thresh, rel_thresh):
    objects = payload.get("objects") or payload.get("detections") or []
    objects = sorted(objects, key=lambda item: item.get("score", item.get("confidence", 0.0)), reverse=True)
    objects = [obj for obj in objects[:box_topk] if obj.get("score", obj.get("confidence", 0.0)) >= box_thresh]
    valid_ids = {obj.get("object_id", obj.get("id")) for obj in objects}

    relations = []
    for rel in payload.get("relations", []):
        if rel.get("subject_id") not in valid_ids or rel.get("object_id") not in valid_ids:
            continue
        if rel.get("score", rel.get("confidence", 0.0)) < rel_thresh:
            continue
        relations.append(rel)
        if len(relations) >= rel_topk:
            break

    return objects, relations


def draw_realtime_image(image_path, objects, relations, output_dir, frame_id):
    if not image_path:
        return None
    if not os.path.exists(image_path):
        print("Realtime image skipped; file not found: {}".format(image_path))
        return None

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    display_size = get_size((orig_w, orig_h))
    img_resized = img.resize(display_size)
    draw = ImageDraw.Draw(img_resized)
    font = get_font(max(12, int(min(display_size) * 0.018)))
    small_font = get_font(max(11, int(min(display_size) * 0.015)))

    scale_w = display_size[0] / orig_w
    scale_h = display_size[1] / orig_h
    centers = {}
    objects_by_id = {obj.get("object_id", obj.get("id")): obj for obj in objects}

    for obj in objects:
        if "bbox" not in obj:
            continue
        x1, y1, x2, y2 = obj["bbox"]
        x1, y1 = x1 * scale_w, y1 * scale_h
        x2, y2 = x2 * scale_w, y2 * scale_h
        object_id = obj.get("object_id", obj.get("id"))
        centers[object_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
        color = PALETTE[object_id % len(PALETTE)]
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)

        label = "{}_{} {:.2f}".format(
            object_id,
            obj.get("label_name", obj.get("label", obj.get("label_id", "object"))),
            obj.get("score", obj.get("confidence", 0.0)),
        )
        if obj.get("center_3d_m"):
            x, y, z = obj["center_3d_m"]
            label += " [{:.2f},{:.2f},{:.2f}m]".format(x, y, z)
        draw_text_label(draw, (x1 + 3, max(2, y1 - 20)), label, font, color)

    for rel in relations:
        if rel["subject_id"] not in centers or rel["object_id"] not in centers:
            continue
        subj = objects_by_id.get(rel["subject_id"])
        color = PALETTE[rel["subject_id"] % len(PALETTE)]
        if rel.get("source") == "geometry":
            color = "orange"
        start = centers[rel["subject_id"]]
        end = centers[rel["object_id"]]
        draw_arrow(draw, start, end, color, 2)
        mid = ((start[0] + end[0]) / 2 + 5, (start[1] + end[1]) / 2 + 5)
        source = rel.get("source", "model")
        text = "{} {} {:.2f}".format(rel.get("predicate", "rel"), source, rel.get("score", 0.0))
        if subj is not None:
            draw_text_label(draw, mid, text, small_font, color)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "realtime_annotated_frame_{}.png".format(frame_id))
    img_resized.save(out_path)
    return out_path


def draw_realtime_graph(objects, relations, output_dir, frame_id):
    G = nx.DiGraph()
    labels = {}
    for obj in objects:
        node = object_node_name(obj)
        G.add_node(node)
        labels[node] = object_graph_label(obj)

    edge_labels = {}
    objects_by_id = {obj.get("object_id", obj.get("id")): obj for obj in objects}
    for rel in relations:
        subj = objects_by_id.get(rel["subject_id"])
        obj = objects_by_id.get(rel["object_id"])
        if not subj or not obj:
            continue
        subj_node = object_node_name(subj)
        obj_node = object_node_name(obj)
        rel_label = "{} {:.2f}".format(rel.get("predicate", "rel"), rel.get("score", rel.get("confidence", 0.0)))
        if rel.get("source"):
            rel_label += "\n{}".format(rel["source"])
        G.add_edge(subj_node, obj_node, label=rel_label)
        edge_labels[(subj_node, obj_node)] = rel_label

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=1.8, seed=42)
        nx.draw(
            G,
            pos,
            labels=labels,
            node_color="#87CEFA",
            node_size=2400,
            edge_color="gray",
            font_size=8,
            font_weight="bold",
            arrows=True,
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)
    plt.title("Scene Graph Knowledge Graph - frame {}".format(frame_id))
    out_path = os.path.join(output_dir, "realtime_knowledge_graph_frame_{}.png".format(frame_id))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def process_realtime_scene_graph(args, db_manager=None):
    with open(args.scene_graph_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frame_id = payload.get("frame_id", "latest")
    objects, relations = filter_realtime_scene_graph(
        payload,
        args.box_topk,
        args.rel_topk,
        args.box_thresh,
        args.rel_thresh,
    )

    print("\n{} 实时场景图 frame: {} {}".format("=" * 20, frame_id, "=" * 20))
    print_list(
        "objects",
        [
            "{}_{}".format(
                obj.get("object_id", obj.get("id")),
                obj.get("label_name", obj.get("label", obj.get("label_id", "object"))),
            )
            for obj in objects
        ],
        [obj.get("score", obj.get("confidence", 0.0)) for obj in objects],
    )
    print_list(
        "relations",
        [
            "{} => {} => {} ({})".format(
                rel.get("subject_id"),
                rel.get("predicate", "rel"),
                rel.get("object_id"),
                rel.get("source", "model"),
            )
            for rel in relations
        ],
        [rel.get("score", rel.get("confidence", 0.0)) for rel in relations],
    )

    annotated_path = draw_realtime_image(args.image_path, objects, relations, args.output_dir, frame_id)
    graph_path = draw_realtime_graph(objects, relations, args.output_dir, frame_id)

    if args.sync_neo4j and db_manager is not None:
        print("Realtime Neo4j sync is not enabled for this JSON format yet; local visualization was generated.")

    if annotated_path:
        print("Saved realtime annotated image: {}".format(annotated_path))
    print("Saved realtime knowledge graph: {}".format(graph_path))


def print_list(name, input_list, scores=None):
    for i, item in enumerate(input_list):
        if scores is None:
            print(f'{name} {i}: {item}')
        else:
            print(f'{name} {i}: {item}; score: {scores[i]}')


def process_single_image(image_idx, custom_prediction, custom_data_info, db_manager):
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']

    # 强制将 JSON 中的相对路径映射为 绝对物理路径
    raw_path = custom_data_info['idx_to_files'][image_idx]
    image_path = raw_path.replace("./input_dir", r"/home/wxm/code/Scene-Graph-Benchmark/input_dir")

    pred = custom_prediction[str(image_idx)]
    all_boxes = pred['bbox'][:BOX_TOPK]
    all_labels_idx = pred['bbox_labels'][:BOX_TOPK]
    box_scores = pred['bbox_scores'][:BOX_TOPK]
    box_labels = [ind_to_classes[idx] for idx in all_labels_idx]

    all_pairs = pred['rel_pairs']
    all_rel_labels = pred['rel_labels']
    all_rel_scores = pred['rel_scores']

    valid_relations = []
    rel_print_labels = []
    rel_print_scores = []
    active_box_indices = set()
    count = 0

    for pair, p_idx, score in zip(all_pairs, all_rel_labels, all_rel_scores):
        s_idx, o_idx = pair[0], pair[1]

        if (
            s_idx < BOX_TOPK
            and o_idx < BOX_TOPK
            and box_scores[s_idx] >= BOX_THRESH
            and box_scores[o_idx] >= BOX_THRESH
            and score >= REL_THRESH
        ):
            predicate = ind_to_predicates[p_idx]
            valid_relations.append({
                'subj_idx': s_idx,
                'obj_idx': o_idx,
                'subj_name': f"{box_labels[s_idx]}_{s_idx}",
                'obj_name': f"{box_labels[o_idx]}_{o_idx}",
                'predicate': predicate,
                'score': score
            })

            label_str = f"{s_idx}_{box_labels[s_idx]} => {predicate} => {o_idx}_{box_labels[o_idx]}"
            rel_print_labels.append(label_str)
            rel_print_scores.append(score)

            active_box_indices.add(s_idx)
            active_box_indices.add(o_idx)

            count += 1

        if count >= REL_TOPK:
            break

    print(f"\n{'=' * 20} 处理图像索引: {image_idx} {'=' * 20}")
    print('*' * 50)
    print_list('box_labels', box_labels, box_scores)
    print('*' * 50)
    print_list('rel_labels', rel_print_labels, rel_print_scores)

    # 同步至 Neo4j 数据库；数据库不可用时不阻塞本地可视化输出
    try:
        db_manager.sync_scene_graph(image_idx, valid_relations, box_labels, box_scores)
    except Exception as exc:
        print(f"Neo4j sync skipped for image {image_idx}: {exc}")

    # 1. 物理空间渲染 (仅边界框)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    display_size = get_size((orig_w, orig_h))
    img_resized = img.resize(display_size)
    draw = ImageDraw.Draw(img_resized)
    font = get_font(max(12, int(min(display_size) * 0.018)))
    small_font = get_font(max(11, int(min(display_size) * 0.015)))

    # 坐标缩放计算：必须恢复缩放逻辑，否则框体会严重位移
    scale_w = display_size[0] / orig_w
    scale_h = display_size[1] / orig_h

    centers = {}
    for i, box in enumerate(all_boxes):
        if i not in active_box_indices:
            continue

        # 应用缩放系数
        x1, y1 = box[0] * scale_w, box[1] * scale_h
        x2, y2 = box[2] * scale_w, box[3] * scale_h
        centers[i] = ((x1 + x2) / 2, (y1 + y2) / 2)
        color = PALETTE[i % len(PALETTE)]

        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
        label_y = max(2, y1 - 20)
        draw_text_label(
            draw,
            (x1 + 3, label_y),
            f"{i}_{box_labels[i]} {box_scores[i]:.2f}",
            font,
            color,
        )

    for rel in valid_relations:
        s_idx = rel["subj_idx"]
        o_idx = rel["obj_idx"]
        if s_idx not in centers or o_idx not in centers:
            continue
        color = PALETTE[s_idx % len(PALETTE)]
        start = centers[s_idx]
        end = centers[o_idx]
        draw_arrow(draw, start, end, color, 2)
        mid = ((start[0] + end[0]) / 2 + 5, (start[1] + end[1]) / 2 + 5)
        draw_text_label(draw, mid, f"{rel['predicate']} {rel['score']:.2f}", small_font, color)

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    image_stem = Path(image_path).stem
    img_resized.save(os.path.join(VIS_OUTPUT_DIR, f'annotated_scene_{image_idx}_{image_stem}.png'))

    # 2. 逻辑空间渲染 (瞬时局部 NetworkX 图)
    G = nx.DiGraph()
    for rel in valid_relations:
        G.add_edge(rel['subj_name'], rel['obj_name'], label=rel['predicate'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#87CEFA', node_size=2000,
            edge_color='gray', font_size=9, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')

    plt.title(f"Scene Graph Knowledge Graph - {image_idx}")
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f'knowledge_graph_final_{image_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    if args.mode in ("realtime", "llm"):
        db_manager = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) if args.sync_neo4j else None
        try:
            process_realtime_scene_graph(args, db_manager)
        finally:
            if db_manager:
                db_manager.close()
        return

    global VIS_OUTPUT_DIR, BOX_TOPK, REL_TOPK, BOX_THRESH, REL_THRESH
    VIS_OUTPUT_DIR = args.output_dir
    BOX_TOPK = args.box_topk
    REL_TOPK = args.rel_topk
    BOX_THRESH = args.box_thresh
    REL_THRESH = args.rel_thresh

    with open(args.pred_path, 'r', encoding='utf-8') as f:
        custom_prediction = json.load(f)
    with open(args.info_path, 'r', encoding='utf-8') as f:
        custom_data_info = json.load(f)

    # 初始化图数据库管理器
    db_manager = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        total_images = len(custom_data_info['idx_to_files'])
        for idx in range(total_images):
            process_single_image(idx, custom_prediction, custom_data_info, db_manager)
    finally:
        # 释放数据库连接资源
        db_manager.close()


if __name__ == '__main__':
    main()
