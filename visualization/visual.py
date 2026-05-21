import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from neo4j import GraphDatabase

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 基础路径配置
PRED_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_prediction.json'
INFO_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_data_info.json'

# 图数据库连接配置 (按实际部署环境修改)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 截断超参数
BOX_TOPK = 30
REL_TOPK = 20


class Neo4jManager:
    """图数据库连接池与写入管理器"""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def sync_scene_graph(self, image_idx, valid_relations, box_labels, box_scores):
        """将单帧场景图写入持久化图数据库"""
        with self.driver.session() as session:
            for rel in valid_relations:
                session.execute_write(self._merge_relation, image_idx, rel, box_labels, box_scores)

    @staticmethod
    def _merge_relation(tx, image_idx, rel, box_labels, box_scores):
        # 构建图谱全局唯一节点ID (基于当前图像索引与实体索引，暂代MOT追踪ID)
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


def print_list(name, input_list, scores=None):
    for i, item in enumerate(input_list):
        if scores is None:
            print(f'{name} {i}: {item}')
        else:
            print(f'{name} {i}: {item}; score: {scores[i]}')


def process_single_image(image_idx, custom_prediction, custom_data_info, db_manager):
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    image_path = custom_data_info['idx_to_files'][image_idx]

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

        if s_idx < BOX_TOPK and o_idx < BOX_TOPK:
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

    # 同步至 Neo4j 数据库
    db_manager.sync_scene_graph(image_idx, valid_relations, box_labels, box_scores)

    # 1. 物理空间渲染 (仅边界框)
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    display_size = get_size((orig_w, orig_h))
    img_resized = img.resize(display_size)
    draw = ImageDraw.Draw(img_resized)

    # 删除或注释掉这两行计算
    # scale_w = display_size[0] / orig_w
    # scale_h = display_size[1] / orig_h

    for i, box in enumerate(all_boxes):
        if i not in active_box_indices:
            continue

        # 2. 修复点：直接使用 JSON 提取的原始坐标，不再乘以 scale 系数
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)
        draw.text((x1, y1), f"{i}_{box_labels[i]}", fill='white')

    img_resized.save(f'annotated_scene_{image_idx}.png')

    # 逻辑空间渲染 (瞬时局部 NetworkX 图)
    G = nx.DiGraph()
    for rel in valid_relations:
        G.add_edge(rel['subj_name'], rel['obj_name'], label=rel['predicate'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#87CEFA', node_size=2000,
            edge_color='gray', font_size=9, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')

    plt.title(f"Scene Graph Knowledge Graph - {image_idx}")
    plt.savefig(f'knowledge_graph_final_{image_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        custom_prediction = json.load(f)
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
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