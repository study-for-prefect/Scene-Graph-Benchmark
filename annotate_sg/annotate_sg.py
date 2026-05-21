import json
import cv2
import numpy as np
import os
import glob


def parse_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = []
    for i, shape in enumerate(data['shapes']):
        points = np.array(shape['points'])
        xmin, ymin = np.min(points, axis=0).astype(int)
        xmax, ymax = np.max(points, axis=0).astype(int)
        objects.append({
            "object_id": i,
            "name": shape['label'],
            "box": [int(xmin), int(ymin), int(xmax), int(ymax)]
        })
    return data.get('imagePath', ''), objects


def annotate_dataset(json_dir, image_dir):
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
            except Exception as e:
                print(f"历史标注文件损坏，重新初始化: {e}")
                relationships = []
        else:
            relationships = []

        global selected_sub, selected_obj
        selected_sub, selected_obj = -1, -1

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
        print("操作: 左键点主体、客体，按键赋谓词。按ESC完成当前图。")
        print(
            "键位映射: [w]上/above [s]下/under [a]左/left_of [d]右/right_of [f]前/in_front_of [b]后/behind [g]抓取/grasping [r]重置选择 [u]撤销")

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

            # 1. 渲染已标注的关系历史
            y_offset = int(30 * scale_factor)
            cv2.putText(vis, f"Saved Relations: {len(relationships)}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        f_scale * 1.2, (0, 165, 255), f_thick)

            for rel in relationships:
                sub_id = rel['subject_id']
                obj_id = rel['object_id']
                pred = rel['predicate']

                y_offset += int(35 * scale_factor)
                list_text = f"[{sub_id}] - {pred} -> [{obj_id}]"
                cv2.putText(vis, list_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 165, 255), f_thick)

                # 中心点有向箭头与谓词标注
                if sub_id in centers and obj_id in centers:
                    pt1 = centers[sub_id]
                    pt2 = centers[obj_id]
                    cv2.arrowedLine(vis, pt1, pt2, (0, 165, 255), arrow_thick, tipLength=0.08)
                    mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_w, text_h), baseline = cv2.getTextSize(pred, font, f_scale, f_thick)

                    # 计算背景矩形坐标，加入动态边距
                    bg_tl = (mid_x - pad, mid_y - text_h - pad)
                    bg_br = (mid_x + text_w + pad, mid_y + baseline)

                    cv2.rectangle(vis, bg_tl, bg_br, (0, 0, 0), -1)
                    cv2.putText(vis, pred, (mid_x, mid_y), font, f_scale, (255, 255, 255), f_thick)

            # 2. 渲染物体边界框与选中状态
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
                break

            if selected_sub != -1 and selected_obj != -1:
                pred = None

                if key == ord('w'):
                    pred = 'above'
                elif key == ord('s'):
                    pred = 'under'
                elif key == ord('a'):
                    pred = 'left_of'
                elif key == ord('d'):
                    pred = 'right_of'
                elif key == ord('f'):
                    pred = 'in_front_of'
                elif key == ord('b'):
                    pred = 'behind'
                elif key == ord('g'):
                    pred = 'grasping'
                elif key == ord('r'):
                    selected_sub, selected_obj = -1, -1
                    print("选择已重置")
                    continue
                elif key == ord('u'):
                    if len(relationships) > 0:
                        removed = relationships.pop()
                        print(
                            f"已撤销关系: [{removed['subject_id']}] - {removed['predicate']} -> [{removed['object_id']}]")
                    else:
                        print("当前图片无关系可撤销")
                    selected_sub, selected_obj = -1, -1
                    continue

                if pred:
                    relationships.append({
                        "subject_id": selected_sub,
                        "object_id": selected_obj,
                        "predicate": pred
                    })
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
    JSON_DIRECTORY = "D:/label/labelme_img/building_block/data"
    IMAGE_DIRECTORY = "D:/label/labelme_img/building_block/data"
    annotate_dataset(JSON_DIRECTORY, IMAGE_DIRECTORY)