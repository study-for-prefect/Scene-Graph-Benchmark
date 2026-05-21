# 引入置信度拦截，避免低分关系被过滤掉

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PRED_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_prediction.json'
INFO_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_data_info.json'

# 1. 替换参数定义区
BOX_TOPK = 50        # 放宽物理截断上限，避免高分实体排序靠后被漏算
REL_TOPK = 50
BOX_THRESH = 0.15    # 新增：实体框置信度下限
REL_THRESH = 0.30    # 新增：关系预测置信度下限


def get_size(image_size):
    # 保持原有函数逻辑不变
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
    # 保持原有函数逻辑不变
    for i, item in enumerate(input_list):
        if scores is None:
            print(f'{name} {i}: {item}')
        else:
            print(f'{name} {i}: {item}; score: {scores[i]}')


def process_single_image(image_idx, custom_prediction, custom_data_info):
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

    # 2. 替换数据过滤循环
    valid_relations = []
    rel_print_labels = []
    rel_print_scores = []
    active_box_indices = set()  # 记录存活实体的索引，用于清理图像冗余框
    count = 0

    for pair, p_idx, score in zip(all_pairs, all_rel_labels, all_rel_scores):
        # 触发关系置信度拦截
        if score < REL_THRESH:
            continue

        s_idx, o_idx = pair[0], pair[1]

        if s_idx < BOX_TOPK and o_idx < BOX_TOPK:
            # 触发主宾实体置信度拦截
            if box_scores[s_idx] < BOX_THRESH or box_scores[o_idx] < BOX_THRESH:
                continue

            predicate = ind_to_predicates[p_idx]
            valid_relations.append({
                'subj_idx': s_idx,
                'obj_idx': o_idx,
                'subj_name': f"{box_labels[s_idx]}_{s_idx}",
                'obj_name': f"{box_labels[o_idx]}_{o_idx}",
                'predicate': predicate
            })

            label_str = f"{s_idx}_{box_labels[s_idx]} => {predicate} => {o_idx}_{box_labels[o_idx]}"
            rel_print_labels.append(label_str)
            rel_print_scores.append(score)

            # 登记存活关系中的实体索引
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

    img = Image.open(image_path)
    orig_w, orig_h = img.size
    display_size = get_size((orig_w, orig_h))
    img_resized = img.resize(display_size)
    draw = ImageDraw.Draw(img_resized)

    scale_w = display_size[0] / orig_w
    scale_h = display_size[1] / orig_h

    centers = {}
    for i, box in enumerate(all_boxes):
        # 隐藏低分及未参与连线的孤立框
        if i not in active_box_indices:
            continue

        x1, y1, x2, y2 = box[0] * scale_w, box[1] * scale_h, box[2] * scale_w, box[3] * scale_h
        draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)
        draw.text((x1, y1), f"{i}_{box_labels[i]}", fill='white')
        centers[i] = ((x1 + x2) / 2, (y1 + y2) / 2)

    # 删除关系连线与文本绘制逻辑
    # for rel in valid_relations:
    #     s_p = centers[rel['subj_idx']]
    #     o_p = centers[rel['obj_idx']]
    #     draw.line([s_p, o_p], fill='lime', width=2)
    #     draw.text(((s_p[0]+o_p[0])/2, (s_p[1]+o_p[1])/2), rel['predicate'], fill='cyan')

    # 动态命名原图标注文件
    img_resized.save(f'annotated_scene_{image_idx}.png')

    G = nx.DiGraph()
    for rel in valid_relations:
        G.add_edge(rel['subj_name'], rel['obj_name'], label=rel['predicate'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#87CEFA', node_size=2000,
            edge_color='gray', font_size=9, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')

    plt.title(f"Scene Graph Knowledge Graph - {image_idx}")
    # 动态命名知识图谱文件，并关闭当前画布以防内存泄漏重叠
    plt.savefig(f'knowledge_graph_final_{image_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        custom_prediction = json.load(f)
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
        custom_data_info = json.load(f)

    # 动态获取 JSON 中包含的图片总数并遍历执行
    total_images = len(custom_data_info['idx_to_files'])
    for idx in range(total_images):
        process_single_image(idx, custom_prediction, custom_data_info)


if __name__ == '__main__':
    main()