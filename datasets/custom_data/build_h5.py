import os
import json
import h5py
import glob
import numpy as np
import re
from PIL import Image

DATA_DIR = "/home/wxm/code/Scene-Graph-Benchmark/datasets/custom_data/data"
sg_files = glob.glob(os.path.join(DATA_DIR, "*_sg.json"))

# 1. 强制且唯一的硬编码字典赋值（置于全局顶部，禁止后续出现任何 label_to_idx = 的赋值操作）
label_to_idx = {
    "__background__": 0, "circle": 1, "rectangle": 2,
    "square blue": 3, "square red": 4, "square yellow": 5,
    "square green": 6, "semi square": 7, "triangle": 8,
    "semi circle": 9, "workspace": 10
}

predicate_to_idx = {
    "__background__": 0, "touching": 1, "stacked_on": 2, "on_surface": 3
}

# 2. 生成底层所需的架构映射字典
with open('custom_dict.json', 'w') as f:
    json.dump({
        'label_to_idx': label_to_idx,
        'predicate_to_idx': predicate_to_idx,
        'attribute_to_idx': {'__background__': 0}
    }, f)



boxes_1024, labels_array = [], []
relationships_array, predicates_array = [], []  # 拆分为独立的边数组和标签数组
img_to_first_box, img_to_last_box = [], []
img_to_first_rel, img_to_last_rel = [], []
b_idx, r_idx = 0, 0
# 新增：图像特征映射表
image_data = []

for f in sg_files:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取并计算真实物理尺度
    img_name = data['image_path']
    base_name = img_name.split('.')[0]
    img_path = os.path.join(DATA_DIR, img_name)

    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except FileNotFoundError:
        h, w = 720, 1280

    # 写入框架兼容的底层索引结构
    image_data.append({
        "image_id": base_name,
        "file_name": img_name,
        "width": w,
        "height": h
    })

    objs = data['objects']
    rels = data.get('relationships', [])

    # 记录当前图像目标框的起始与终止物理索引
    img_to_first_box.append(b_idx)
    if len(objs) > 0:
        img_to_last_box.append(b_idx + len(objs) - 1)
    else:
        img_to_last_box.append(-1)
    b_idx += len(objs)

    # 强制正则清洗拦截块
    for obj in objs:
        # visual_genome.py expects boxes_1024 as scaled (cx, cy, w, h).
        # The annotation json stores absolute (x1, y1, x2, y2) image coordinates.
        bx = obj['box']
        x1, y1 = min(bx[0], bx[2]), min(bx[1], bx[3])
        x2, y2 = max(bx[0], bx[2]), max(bx[1], bx[3])
        scale = 1024.0 / max(w, h)
        boxes_1024.append([
            ((x1 + x2) / 2.0) * scale,
            ((y1 + y2) / 2.0) * scale,
            (x2 - x1) * scale,
            (y2 - y1) * scale,
        ])

        raw_name = str(obj['name']).lower()
        clean_name = re.sub(r'[^a-z ]', '', raw_name).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name)

        if clean_name not in label_to_idx:
            raise ValueError(f"致命异常：字典未映射 [{clean_name}]")

        labels_array.append([label_to_idx[clean_name]])

    if len(rels) > 0:
        img_to_first_rel.append(r_idx)
        img_to_last_rel.append(r_idx + len(rels) - 1)
        r_idx += len(rels)
        for rel in rels:
            # 强制将局部索引加上首框偏移量，转换为底层框架要求的全局绝对索引
            relationships_array.append([
                rel['subject_id'] + img_to_first_box[-1],
                rel['object_id'] + img_to_first_box[-1]
            ])
            if rel['predicate'] not in predicate_to_idx:
                raise ValueError(f"致命异常：谓词字典未映射 [{rel['predicate']}]，文件: {f}")

            # predicates_array 仅记录 [谓词索引]
            predicates_array.append([predicate_to_idx[rel['predicate']]])
    else:
        img_to_first_rel.append(-1)
        img_to_last_rel.append(-1)

# 构建 split 数组，长度等于图片总数，全部赋值为 0 (Train)
split_array = np.zeros(len(sg_files), dtype=np.int32)

# 新增：构建 attributes 占位张量，严格对齐二维数组形状
attributes_array = np.zeros((len(boxes_1024), 10), dtype=np.int64)

with h5py.File('custom_data.h5', 'w') as h5_file:
    h5_file.create_dataset('boxes_1024', data=np.array(boxes_1024, dtype=np.float32))
    h5_file.create_dataset('labels', data=np.array(labels_array, dtype=np.int64))
    # 写入拆分后的拓扑和谓词
    h5_file.create_dataset('relationships', data=np.array(relationships_array, dtype=np.int32))
    h5_file.create_dataset('predicates', data=np.array(predicates_array, dtype=np.int32))

    h5_file.create_dataset('img_to_first_box', data=np.array(img_to_first_box, dtype=np.int32))
    h5_file.create_dataset('img_to_last_box', data=np.array(img_to_last_box, dtype=np.int32))
    h5_file.create_dataset('img_to_first_rel', data=np.array(img_to_first_rel, dtype=np.int32))
    h5_file.create_dataset('img_to_last_rel', data=np.array(img_to_last_rel, dtype=np.int32))
    h5_file.create_dataset('split', data=split_array)  # 新增张量：数据集划分标识
    h5_file.create_dataset('attributes', data=attributes_array)  # 追加属性张量
# 写入 FPN 所需的底层图像映射文件
with open('custom_image_data.json', 'w') as out:
    json.dump(image_data, out)
