import json
import numpy as np
import os
import glob


LABEL_TO_IDX = {
    "background": 0,
    "__background__": 0,
    "circle": 1,
    "rectangle": 2,
    "square blue": 3,
    "square red": 4,
    "square yellow": 5,
    "square green": 6,
    "semi square": 7,
    "triangle": 8,
    "semi circle": 9,
    "workspace": 10,
}

PREDICATE_KEYS = {
    ord('t'): "touching",
    ord('s'): "stacked_on",
    ord('o'): "on_surface",
}

WORKSPACE_LABEL = "workspace"
BLOCK_LABELS = set(LABEL_TO_IDX) - {"background", "__background__", WORKSPACE_LABEL}


def normalize_label(label):
    return " ".join(str(label).lower().strip().split())


def parse_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = []
    for i, shape in enumerate(data['shapes']):
        name = normalize_label(shape['label'])
        points = np.array(shape['points'])
        xmin, ymin = np.min(points, axis=0).astype(int)
        xmax, ymax = np.max(points, axis=0).astype(int)
        if name not in LABEL_TO_IDX:
            print(f"警告: 未在字典中登记的类别 [{name}]，文件: {json_path}")
        objects.append({
            "object_id": i,
            "name": name,
            "box": [int(xmin), int(ymin), int(xmax), int(ymax)]
        })
    return data.get('imagePath', ''), objects


def is_block(obj):
    return obj["name"] in BLOCK_LABELS


def is_workspace(obj):
    return obj["name"] == WORKSPACE_LABEL


def object_by_id(objects, object_id):
    for obj in objects:
        if obj["object_id"] == object_id:
            return obj
    return None


def center_of(obj):
    bx1, by1, bx2, by2 = obj["box"]
    return ((bx1 + bx2) // 2, (by1 + by2) // 2)


def has_duplicate_relation(relationships, sub_id, obj_id, predicate, ignore_index=None):
    for idx, rel in enumerate(relationships):
        if idx == ignore_index:
            continue
        if rel["predicate"] != predicate:
            continue
        if predicate == "touching":
            if {rel["subject_id"], rel["object_id"]} == {sub_id, obj_id}:
                return True
        elif rel["subject_id"] == sub_id and rel["object_id"] == obj_id:
            return True
    return False


def validate_relation(objects, relationships, sub_id, obj_id, predicate, ignore_index=None):
    sub = object_by_id(objects, sub_id)
    obj = object_by_id(objects, obj_id)
    if sub is None or obj is None:
        return False, "主体或客体不存在"
    if sub_id == obj_id:
        return False, "主体和客体不能是同一个对象"
    if has_duplicate_relation(relationships, sub_id, obj_id, predicate, ignore_index=ignore_index):
        return False, f"重复关系: [{sub_id}] - {predicate} -> [{obj_id}]"

    if predicate == "on_surface":
        if not is_block(sub) or not is_workspace(obj):
            return False, "on_surface 必须是: 积木 -> workspace"
    elif predicate == "stacked_on":
        if not is_block(sub) or not is_block(obj):
            return False, "stacked_on 必须是: 上方积木 -> 下方积木"
    elif predicate == "touching":
        if not is_block(sub) or not is_block(obj):
            return False, "touching 必须是: 积木 -> 积木"
        sub_center = center_of(sub)
        obj_center = center_of(obj)
        if sub_center[0] > obj_center[0]:
            return False, "touching 方向建议固定为: 左侧积木 -> 右侧积木"
    else:
        return False, f"未知谓词: {predicate}"

    return True, ""


def print_image_checks(objects, relationships):
    workspace_ids = [obj["object_id"] for obj in objects if is_workspace(obj)]
    if not workspace_ids:
        print("警告: 当前图没有 workspace，无法完整标注 on_surface。")
        return

    stacked_subjects = {
        rel["subject_id"]
        for rel in relationships
        if rel["predicate"] == "stacked_on"
    }
    blocks = [
        obj for obj in objects
        if is_block(obj) and obj["object_id"] not in stacked_subjects
    ]
    surface_subjects = {
        rel["subject_id"]
        for rel in relationships
        if rel["predicate"] == "on_surface" and rel["object_id"] in workspace_ids
    }
    missing = [obj for obj in blocks if obj["object_id"] not in surface_subjects]
    if missing:
        names = ", ".join(f"{obj['object_id']}:{obj['name']}" for obj in missing)
        print(f"警告: 以下接地积木还没有 on_surface -> workspace: {names}")


def clamp_relation_index(index, relationships):
    if not relationships:
        return -1
    if index < 0:
        return len(relationships) - 1
    if index >= len(relationships):
        return 0
    return index


def annotate_dataset(json_dir, image_dir):
    import cv2

    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files = [f for f in json_files if not f.endswith('_sg.json')]

    for json_path in json_files:
        img_filename, objects = parse_labelme_json(json_path)
        if not img_filename:
            img_filename = os.path.basename(json_path).replace('.json', '.jpg')

        img_path = os.path.join(image_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        out_name = json_path.replace('.json', '_sg.json')
        if os.path.exists(out_name):
            try:
                with open(out_name, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                relationships = history_data.get('relationships', [])
                print(f"已载入历史标注关系: {len(relationships)} 条")
            except Exception:
                print(f"历史标注文件损坏，重新初始化。")
                relationships = []
        else:
            relationships = []

        global selected_sub, selected_obj
        selected_sub, selected_obj = -1, -1
        selected_rel_idx = -1

        def mouse_click(event, x, y, flags, param):
            global selected_sub, selected_obj
            if event == cv2.EVENT_LBUTTONDOWN:
                for obj in objects:
                    bx1, by1, bx2, by2 = obj['box']
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        if selected_sub == -1:
                            selected_sub = obj['object_id']
                            print(f"[主体] {obj['name']} (ID:{selected_sub})")
                        elif selected_obj == -1 and obj['object_id'] != selected_sub:
                            selected_obj = obj['object_id']
                            print(f"[客体] {obj['name']} (ID:{selected_obj}) -> 等待谓词输入")
                        break

        cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotation', 1280, 720)
        cv2.setMouseCallback('Annotation', mouse_click)

        print(f"\n当前处理: {img_filename}")
        print("操作: 左键点主体、客体，按键赋谓词。按ESC保存当前图。")
        print("键位映射: [t] touching  [s] stacked_on  [o] on_surface  [r]重置  [u]撤销")
        print("修改关系: [n]/[p]选择已有关系  [x]删除选中关系  [e]退出修改模式")
        print("替换关系: 先用[n]/[p]选中关系，再左键点新主体、客体，按[t]/[s]/[o]替换。")
        print("规则: on_surface = 积木 -> workspace；stacked_on = 上方积木 -> 下方积木；touching = 左侧积木 -> 右侧积木，仅标一次。")

        centers = {}
        for obj in objects:
            bx1, by1, bx2, by2 = obj['box']
            centers[obj['object_id']] = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        img_h, img_w = img.shape[:2]
        scale_factor = max(img_h, img_w) / 1000.0

        f_scale = max(0.6, 0.8 * scale_factor)
        f_thick = max(1, int(2 * scale_factor))
        box_thick = max(1, int(2 * scale_factor))
        arrow_thick = max(2, int(3 * scale_factor))
        pad = int(4 * scale_factor)

        while True:
            vis = img.copy()

            y_offset = int(30 * scale_factor)
            mode_text = "ADD" if selected_rel_idx == -1 else f"EDIT #{selected_rel_idx}"
            cv2.putText(vis, f"Saved Relations: {len(relationships)}  Mode: {mode_text}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        f_scale * 1.2, (0, 165, 255), f_thick)

            for rel_idx, rel in enumerate(relationships):
                sub_id = rel['subject_id']
                obj_id = rel['object_id']
                pred = rel['predicate']
                is_selected_rel = rel_idx == selected_rel_idx
                rel_color = (0, 255, 255) if is_selected_rel else (0, 165, 255)
                rel_thick = f_thick * 2 if is_selected_rel else f_thick

                y_offset += int(35 * scale_factor)
                list_text = f"{rel_idx}: [{sub_id}] - {pred} -> [{obj_id}]"
                cv2.putText(vis, list_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, f_scale, rel_color, rel_thick)

                if sub_id in centers and obj_id in centers:
                    pt1 = centers[sub_id]
                    pt2 = centers[obj_id]
                    cv2.arrowedLine(vis, pt1, pt2, rel_color, arrow_thick + (2 if is_selected_rel else 0), tipLength=0.08)
                    mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_w, text_h), baseline = cv2.getTextSize(pred, font, f_scale, f_thick)
                    bg_tl = (mid_x - pad, mid_y - text_h - pad)
                    bg_br = (mid_x + text_w + pad, mid_y + baseline)

                    cv2.rectangle(vis, bg_tl, bg_br, (0, 0, 0), -1)
                    cv2.putText(vis, pred, (mid_x, mid_y), font, f_scale, (255, 255, 255), f_thick)

            for obj in objects:
                bx1, by1, bx2, by2 = obj['box']
                color = (0, 255, 0)
                thickness = box_thick

                if obj['object_id'] == selected_sub:
                    color = (0, 0, 255)
                    thickness = box_thick * 2
                elif obj['object_id'] == selected_obj:
                    color = (255, 0, 0)
                    thickness = box_thick * 2

                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, thickness)
                cv2.putText(vis, f"{obj['object_id']}:{obj['name']}", (bx1, by1 - pad),
                            cv2.FONT_HERSHEY_SIMPLEX, f_scale, color, thickness)

            cv2.imshow('Annotation', vis)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                print_image_checks(objects, relationships)
                break

            if key == ord('r'):
                selected_sub, selected_obj = -1, -1
                print("选择已重置")
                continue

            if key == ord('e'):
                selected_rel_idx = -1
                selected_sub, selected_obj = -1, -1
                print("已退出修改模式，后续关系将追加写入")
                continue

            if key == ord('n'):
                if relationships:
                    selected_rel_idx = clamp_relation_index(selected_rel_idx + 1, relationships)
                    rel = relationships[selected_rel_idx]
                    print(f"选中关系 #{selected_rel_idx}: [{rel['subject_id']}] - {rel['predicate']} -> [{rel['object_id']}]")
                else:
                    print("当前图片没有可选择的关系")
                selected_sub, selected_obj = -1, -1
                continue

            if key == ord('p'):
                if relationships:
                    selected_rel_idx = clamp_relation_index(selected_rel_idx - 1, relationships)
                    rel = relationships[selected_rel_idx]
                    print(f"选中关系 #{selected_rel_idx}: [{rel['subject_id']}] - {rel['predicate']} -> [{rel['object_id']}]")
                else:
                    print("当前图片没有可选择的关系")
                selected_sub, selected_obj = -1, -1
                continue

            if key == ord('x'):
                if selected_rel_idx != -1 and relationships:
                    removed = relationships.pop(selected_rel_idx)
                    print(f"已删除关系 #{selected_rel_idx}: [{removed['subject_id']}] - {removed['predicate']} -> [{removed['object_id']}]")
                    selected_rel_idx = clamp_relation_index(selected_rel_idx, relationships)
                else:
                    print("请先用 [n]/[p] 选中要删除的关系")
                selected_sub, selected_obj = -1, -1
                continue

            if key == ord('u'):
                if len(relationships) > 0:
                    removed = relationships.pop()
                    print(f"已撤销关系: [{removed['subject_id']}] - {removed['predicate']} -> [{removed['object_id']}]")
                    selected_rel_idx = clamp_relation_index(selected_rel_idx, relationships)
                else:
                    print("当前图片无关系可撤销")
                selected_sub, selected_obj = -1, -1
                continue

            if selected_sub != -1 and selected_obj != -1:
                pred = PREDICATE_KEYS.get(key)

                if pred:
                    ignore_index = selected_rel_idx if selected_rel_idx != -1 else None
                    ok, message = validate_relation(
                        objects,
                        relationships,
                        selected_sub,
                        selected_obj,
                        pred,
                        ignore_index=ignore_index
                    )
                    if not ok:
                        print(f"拒绝写入: {message}")
                        selected_sub, selected_obj = -1, -1
                        continue
                    new_rel = {
                        "subject_id": selected_sub,
                        "object_id": selected_obj,
                        "predicate": pred
                    }
                    if selected_rel_idx != -1 and relationships:
                        old_rel = relationships[selected_rel_idx]
                        relationships[selected_rel_idx] = new_rel
                        print(
                            f"已替换关系 #{selected_rel_idx}: "
                            f"[{old_rel['subject_id']}] - {old_rel['predicate']} -> [{old_rel['object_id']}] "
                            f"=> [{selected_sub}] - {pred} -> [{selected_obj}]"
                        )
                    else:
                        relationships.append(new_rel)
                        print(f"写入关系: [{selected_sub}] - {pred} -> [{selected_obj}]")
                    selected_sub, selected_obj = -1, -1

        output_data = {
            "image_path": img_filename,
            "objects": objects,
            "relationships": relationships
        }
        out_name = json_path.replace('.json', '_sg.json')
        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    JSON_DIRECTORY = "/home/wxm/dataset/building_block"
    IMAGE_DIRECTORY = "/home/wxm/dataset/building_block"
    annotate_dataset(JSON_DIRECTORY, IMAGE_DIRECTORY)
