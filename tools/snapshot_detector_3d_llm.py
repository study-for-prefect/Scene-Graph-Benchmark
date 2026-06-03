"""Single-frame detector + 3D coordinates + one-LLM scene graph/decision.

Pipeline:
1. Capture one aligned RGB-D frame from RealSense and release the camera.
2. Run the detector model on the RGB snapshot.
3. Attach 3D camera coordinates from the aligned depth frame.
4. Send objects + 3D coordinates + instruction to one Ollama model.
"""

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests


DEFAULT_OUT = "/tmp/snapshot_detector_3d_llm"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "custom_output_detector", "config.yml")
DEFAULT_WEIGHT = os.path.join(PROJECT_ROOT, "custom_output_detector", "model_final.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Snapshot detector + 3D coordinates + Qwen reasoning.")
    parser.add_argument("--instruction", default="描述当前场景，并给出下一步安全操作建议")
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--image-in", default="", help="Use an existing color image instead of capturing from RealSense.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=30, help="Maximum filtered detections sent to the LLM.")
    parser.add_argument("--detector-scale", type=float, default=1.0, help="Upscale the snapshot before detector inference, then map boxes back.")
    parser.add_argument("--debug-topk", type=int, default=30, help="Save top raw detector candidates before score filtering.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--auto-profile", action="store_true", default=True, help="Try fallback RGB-D stream profiles if the requested one times out.")
    parser.add_argument("--no-auto-profile", action="store_false", dest="auto_profile")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--rs-reset", action="store_true")
    parser.add_argument("--rs-exposure", type=float, default=-1.0)
    parser.add_argument("--rs-gain", type=float, default=-1.0)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--no-image", action="store_true", help="Do not send the snapshot image to the VLM.")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--compile-execution-plan", action="store_true", help="Also compile a coordinate execution plan after LLM output.")
    parser.add_argument("--place-offset-m", type=float, default=0.08)
    parser.add_argument("--approach-height-m", type=float, default=0.08)
    parser.add_argument("--show", action="store_true", help="Show annotated detector output.")
    return parser.parse_args()


def project_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        mkdir(parent)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_label_names(output_dir):
    label_file = os.path.join(output_dir, "labels.json")
    if not os.path.exists(label_file):
        return {}
    with open(label_file, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, list):
        return {str(index): name for index, name in enumerate(labels)}
    return labels


def coordinate_convention():
    return {
        "frame": "realsense_color_optical_frame",
        "unit": "meter",
        "x": "positive camera-right, negative camera-left",
        "y": "positive downward in optical frame",
        "z": "positive forward from camera; smaller z is closer/in front of larger z",
    }


def get_field(boxlist, *names):
    for name in names:
        if boxlist.has_field(name):
            return boxlist.get_field(name)
    raise KeyError("None of the requested fields exist: {}".format(names))


def color_for_label(label_id):
    rng = np.random.default_rng(label_id * 9973)
    return tuple(int(v) for v in rng.integers(60, 240, size=3))


def configure_color_sensor(rs, profile, exposure, gain):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        if "rgb" not in name.lower():
            continue
        if exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        elif sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        if exposure >= 0 and sensor.supports(rs.option.exposure):
            sensor.set_option(rs.option.exposure, float(exposure))
        if gain >= 0 and sensor.supports(rs.option.gain):
            sensor.set_option(rs.option.gain, float(gain))
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 1)


def hardware_reset(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found.")
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print("Hardware-resetting RealSense serial={}...".format(serial), flush=True)
    devices[0].hardware_reset()
    time.sleep(8.0)


def list_realsense_devices(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    print("RealSense devices: {}".format(len(devices)), flush=True)
    for idx, dev in enumerate(devices):
        fields = {}
        for info in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.usb_type_descriptor,
            rs.camera_info.physical_port,
        ]:
            if dev.supports(info):
                fields[str(info)] = dev.get_info(info)
        print("device {}: {}".format(idx, fields), flush=True)


def stream_profiles(args):
    requested = (args.width, args.height, args.depth_width, args.depth_height, args.fps)
    candidates = [
        requested,
        (640, 480, 640, 360, 15),
        (640, 480, 480, 270, 15),
        (424, 240, 480, 270, 15),
        (640, 480, 640, 480, 30),
        (640, 480, 640, 360, 30),
        (640, 480, 480, 270, 30),
    ]
    output = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if not args.auto_profile:
            break
    return output


def read_aligned_rgbd(rs, pipeline, align, timeout_ms, warmup_frames):
    frame_bgr = None
    depth_frame = None
    for idx in range(max(0, warmup_frames) + 1):
        frames = pipeline.wait_for_frames(timeout_ms)
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue
        frame_bgr = np.asanyarray(color.get_data())
        depth_frame = depth
        if idx < warmup_frames:
            continue
        break
    if frame_bgr is None or depth_frame is None:
        raise RuntimeError("No aligned RGB-D frame received.")
    return frame_bgr, depth_frame


def capture_color_only(rs, args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    print(
        "Starting RealSense color-only fallback color={}x{}@{}".format(
            args.width, args.height, args.fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    try:
        frame_bgr = None
        for idx in range(max(0, args.warmup_frames) + 1):
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            color = frames.get_color_frame()
            if not color:
                continue
            frame_bgr = np.asanyarray(color.get_data())
            if idx < args.warmup_frames:
                continue
            break
        if frame_bgr is None:
            raise RuntimeError("No color frame received.")
        used_profile = {
            "color_width": args.width,
            "color_height": args.height,
            "depth_width": None,
            "depth_height": None,
            "fps": args.fps,
            "depth_available": False,
        }
        return frame_bgr, None, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)

    print(
        "Starting RealSense color={}x{}@{} depth={}x{}@{}".format(
            width, height, fps, depth_width, depth_height, fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    try:
        frame_bgr, depth_frame = read_aligned_rgbd(
            rs,
            pipeline,
            align,
            args.frame_timeout_ms,
            args.warmup_frames,
        )
        used_profile = {
            "color_width": width,
            "color_height": height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "fps": fps,
            "depth_available": True,
        }
        return frame_bgr, depth_frame, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd(args):
    import pyrealsense2 as rs

    list_realsense_devices(rs)
    if args.rs_reset:
        hardware_reset(rs)

    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        if width < args.width or height < args.height:
            print(
                "Skipping lower color profile {}x{} to preserve detector resolution; will use color-only fallback if needed.".format(
                    width, height
                ),
                flush=True,
            )
            continue
        try:
            return capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps)
        except RuntimeError as exc:
            error = "profile color={}x{}@{} depth={}x{}@{} failed: {}".format(
                width, height, fps, depth_width, depth_height, fps, exc
            )
            print(error, flush=True)
            errors.append(error)
            time.sleep(1.0)
    try:
        return capture_color_only(rs, args)
    except RuntimeError as exc:
        errors.append("color-only fallback failed: {}".format(exc))
    raise RuntimeError("All RealSense profiles failed:\n{}".format("\n".join(errors)))


class DetectorModel:
    def __init__(self, args):
        from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

        import torch
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.data.transforms import build_transforms
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

        self.torch = torch
        self.cfg = cfg
        self.config_file = project_path(args.config_file)
        self.weight = project_path(args.weight)
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHT = self.weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH = min(cfg.MODEL.ROI_HEADS.SCORE_THRESH, args.score_thresh)
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.", flush=True)
            device = "cpu"
        cfg.MODEL.DEVICE = device
        if device == "cpu":
            cfg.DTYPE = "float32"
        cfg.freeze()

        self.device = cfg.MODEL.DEVICE
        self.model = build_detection_model(cfg)
        self.model.to(self.device)
        self.model.eval()
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="")
        checkpointer.load(self.weight, with_optim=False)
        self.transforms = build_transforms(cfg, is_train=False)
        self.size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY
        self.label_names = load_label_names(os.path.dirname(self.weight)) or load_label_names(cfg.OUTPUT_DIR)

    def predict(self, frame_bgr, score_thresh, detector_scale, max_detections):
        from PIL import Image
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        from maskrcnn_benchmark.structures.image_list import to_image_list

        inference_bgr = frame_bgr
        scale = float(detector_scale)
        if scale > 1.0:
            inference_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            scale = 1.0

        image_rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        original_size = pil_image.size
        empty_target = BoxList(self.torch.empty((0, 4)), original_size, mode="xyxy")
        tensor, _ = self.transforms(pil_image, empty_target)
        with self.torch.no_grad():
            image_list = to_image_list([tensor], self.size_divisibility).to(self.device)
            prediction = self.model(image_list)[0].to("cpu")
        prediction = prediction.resize(original_size).convert("xyxy")

        labels = get_field(prediction, "pred_labels", "labels").tolist()
        scores = get_field(prediction, "pred_scores", "scores").tolist()
        boxes = (prediction.bbox / scale).tolist()

        candidates = []
        detections = []
        sorted_items = sorted(zip(boxes, labels, scores), key=lambda item: item[2], reverse=True)
        for rank, (box, label, score) in enumerate(sorted_items):
            label_id = int(label)
            candidate = {
                "rank": rank,
                "bbox": [float(v) for v in box],
                "label_id": label_id,
                "label": self.label_names.get(str(label_id), str(label_id)),
                "confidence": float(score),
            }
            candidates.append(candidate)
            if score < score_thresh:
                continue
            detections.append(
                {
                    "id": len(detections),
                    "bbox": [float(v) for v in box],
                    "label_id": label_id,
                    "label": self.label_names.get(str(label_id), str(label_id)),
                    "confidence": float(score),
                }
            )
            if len(detections) >= max_detections:
                break
        return detections, candidates


def robust_depth(depth_frame, cx, cy, window):
    half = max(0, int(window) // 2)
    values = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    for y in range(max(0, cy - half), min(height, cy + half + 1)):
        for x in range(max(0, cx - half), min(width, cx + half + 1)):
            value = float(depth_frame.get_distance(x, y))
            if value > 0:
                values.append(value)
    if not values:
        return 0.0
    return float(np.median(values))


def clamp_int(value, low, high):
    return max(low, min(high, int(round(value))))


def attach_3d(detections, depth_frame, intrinsics, args):
    if depth_frame is None:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            det["center_px"] = [cx, cy]
            det["depth_m"] = 0.0
            det["center_3d_m"] = None
            det["coordinate_valid"] = False
        return detections

    import pyrealsense2 as rs

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = clamp_int((x1 + x2) / 2.0, 0, depth_width - 1)
        cy = clamp_int((y1 + y2) / 2.0, 0, depth_height - 1)
        depth_m = robust_depth(depth_frame, cx, cy, args.depth_window)
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], float(depth_m)) if depth_m > 0 else None
        det["center_px"] = [cx, cy]
        det["depth_m"] = depth_m
        det["center_3d_m"] = [float(v) for v in point] if point is not None else None
        det["coordinate_valid"] = point is not None
    return detections


def draw_annotated(frame_bgr, detections):
    output = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = color_for_label(det["label_id"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = "{}:{} {:.2f}".format(det["id"], det["label"], det["confidence"])
        if det.get("center_3d_m"):
            x, y, z = det["center_3d_m"]
            text += " xyz[{:.2f},{:.2f},{:.2f}]".format(x, y, z)
        cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output


def public_object(det):
    return {
        "id": det["id"],
        "label": det["label"],
        "confidence": round(det["confidence"], 4),
        "bbox_xyxy_px": [round(v, 2) for v in det["bbox"]],
        "center_px": det.get("center_px"),
        "depth_m": round(det.get("depth_m", 0.0), 4),
        "center_3d_m": [round(v, 4) for v in det["center_3d_m"]] if det.get("center_3d_m") else None,
        "coordinate_valid": bool(det.get("coordinate_valid")),
    }


def build_private_state(args, detections, snapshot_path, annotated_path, used_profile):
    return {
        "schema_version": "private_scene_state_v1",
        "frame_id": "snapshot_{}".format(int(time.time() * 1000)),
        "timestamp": time.time(),
        "instruction": args.instruction,
        "snapshot_image": os.path.abspath(snapshot_path),
        "annotated_image": os.path.abspath(annotated_path),
        "camera_profile": used_profile,
        "coordinate_convention": coordinate_convention(),
        "objects": [
            dict(
                public_object(det),
                label_id=det.get("label_id"),
                is_workspace=det.get("label") == "workspace",
            )
            for det in detections
        ],
    }


def build_llm_input(args, detections, snapshot_path, used_profile):
    return {
        "schema_version": "detector_3d_llm_input_v1",
        "instruction": args.instruction,
        "visual_context": {
            "snapshot_image": snapshot_path,
            "usage": "The attached original RGB image is global visual context. Use it to verify layout and appearance, but do not invent objects that are absent from hard_priors.objects.",
        },
        "camera_profile": used_profile,
        "hard_priors": {
            "description": "Detector boxes/classes/confidences and RealSense depth-derived 3D centers. Treat these as primary structured evidence for relation and action decisions.",
            "coordinate_convention": coordinate_convention(),
            "objects": [public_object(det) for det in detections],
        },
    }


def build_prompt(llm_input):
    return """你是机器人场景理解和任务决策模块。

输入由两部分组成：
1. 原始 RGB 图片：作为全局上下文视觉特征，用来辅助核对场景布局、遮挡和外观。
2. Hard Prior 文本：目标检测类别、置信度、bbox、深度值和每个物体中心点的 3D 坐标。关系判断和动作目标必须优先依据这些硬先验。

空间关系规则：
- 只允许引用 hard_priors.objects 中已有的 object id，不要编造不存在的物体。
- x 更小表示更靠左，x 更大表示更靠右。
- z 更小表示更靠近相机/in_front_of，z 更大表示更远/behind。
- near/far 需要结合 3D 欧氏距离判断。
- on_surface 只有在有 workspace/table 类参照物，且图片上下文与 bbox/3D 坐标共同支持时才输出；不确定则输出 unknown。
- center_3d_m 为 null 或 coordinate_valid 为 false 的对象，关系只能保守判断，并把原因写入 uncertainties。

请先生成场景图，再结合用户指令生成给后续执行模块使用的动作计划。
不要输出关节角、速度、底层电机命令或 ROS topic。若指令无法安全执行，使用 ask_user 或 stop。

请只输出严格 JSON，schema 如下：
{
  "scene_graph": {
    "objects": [{"id": 0, "label": "object name", "confidence": 0.95}],
    "relations": [
      {
        "subject_id": 0,
        "predicate": "left_of|right_of|in_front_of|behind|near|far|on_surface|unknown",
        "object_id": 1,
        "reason": "short coordinate-based reason"
      }
    ],
    "summary": "short scene summary"
  },
  "action_plan": [
    {
      "step": 1,
      "action": "move_named_pose|open_gripper|close_gripper|pick|place_relative|move_above|ask_user|stop",
      "object_id": null,
      "reference_object_id": null,
      "named_pose": null,
      "relative_position": "left_of|right_of|in_front_of|behind|on_surface|near|center_of_workspace|null",
      "target_position_3d_m": null,
      "reason": "short reason"
    }
  ],
  "safety_checks": [],
  "uncertainties": []
}

输入：
{}""".format(json.dumps(llm_input, ensure_ascii=False, indent=2))


def call_ollama(args, prompt, snapshot_path):
    message = {"role": "user", "content": prompt}
    if not args.no_image:
        message["images"] = [image_to_base64(snapshot_path)]
    payload = {
        "model": args.model,
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": args.num_predict},
    }
    response = requests.post(args.ollama_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", json.dumps(data, ensure_ascii=False))


def parse_json_or_embedded(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def compile_execution_plan(args, decision_text, private_state, output_path):
    from types import SimpleNamespace

    from tools.decision_to_execution import compile_plan

    decision = parse_json_or_embedded(decision_text)
    if "action_plan" not in decision and isinstance(decision.get("task_decision"), dict):
        step = dict(decision["task_decision"])
        step.setdefault("step", 1)
        decision["action_plan"] = [step]
    plan_args = SimpleNamespace(
        place_offset_m=args.place_offset_m,
        approach_height_m=args.approach_height_m,
    )
    plan = compile_plan(decision, private_state, plan_args)
    write_json(output_path, plan)
    return plan


def main():
    args = parse_args()
    args.output_dir = project_path(args.output_dir)
    args.config_file = project_path(args.config_file)
    args.weight = project_path(args.weight)
    mkdir(args.output_dir)
    snapshot_path = os.path.join(args.output_dir, "snapshot.jpg")
    annotated_path = os.path.join(args.output_dir, "annotated_detector.jpg")
    objects_path = os.path.join(args.output_dir, "detector_objects_3d.json")
    candidates_path = os.path.join(args.output_dir, "detector_candidates.json")
    llm_input_path = os.path.join(args.output_dir, "llm_input.json")
    private_state_path = os.path.join(args.output_dir, "private_scene_state.json")
    decision_path = os.path.join(args.output_dir, "llm_scene_graph_decision.json")
    execution_plan_path = os.path.join(args.output_dir, "robot_execution_plan.json")

    if args.image_in:
        args.image_in = project_path(args.image_in)
        frame_bgr = cv2.imread(args.image_in, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError("Failed to read image: {}".format(args.image_in))
        depth_frame = None
        intrinsics = None
        used_profile = {
            "color_width": int(frame_bgr.shape[1]),
            "color_height": int(frame_bgr.shape[0]),
            "depth_width": None,
            "depth_height": None,
            "fps": None,
            "depth_available": False,
            "source": args.image_in,
        }
    else:
        frame_bgr, depth_frame, intrinsics, used_profile = capture_rgbd(args)
    cv2.imwrite(snapshot_path, frame_bgr)
    print("Saved snapshot: {}".format(snapshot_path), flush=True)

    detector = DetectorModel(args)
    detections, candidates = detector.predict(
        frame_bgr,
        args.score_thresh,
        args.detector_scale,
        args.max_detections,
    )
    write_json(candidates_path, {"score_threshold": args.score_thresh, "candidates": candidates[: args.debug_topk]})
    print("Saved detector candidates: {}".format(candidates_path), flush=True)
    detections = attach_3d(detections, depth_frame, intrinsics, args)
    write_json(
        objects_path,
        {
            "score_threshold": args.score_thresh,
            "max_detections": args.max_detections,
            "coordinate_convention": coordinate_convention(),
            "objects": detections,
        },
    )

    annotated = draw_annotated(frame_bgr, detections)
    cv2.imwrite(annotated_path, annotated)
    print("Saved annotated detector image: {}".format(annotated_path), flush=True)

    private_state = build_private_state(args, detections, snapshot_path, annotated_path, used_profile)
    write_json(private_state_path, private_state)
    print("Saved private scene state: {}".format(private_state_path), flush=True)

    llm_input = build_llm_input(args, detections, snapshot_path, used_profile)
    write_json(llm_input_path, llm_input)
    print("Saved LLM input: {}".format(llm_input_path), flush=True)

    if not args.skip_llm:
        prompt = build_prompt(llm_input)
        result = call_ollama(args, prompt, snapshot_path)
        with open(decision_path, "w", encoding="utf-8") as f:
            f.write(result)
        print("Saved LLM scene graph and decision: {}".format(decision_path), flush=True)
        print(result)
        if args.compile_execution_plan:
            compile_execution_plan(args, result, private_state, execution_plan_path)
            print("Saved robot execution plan: {}".format(execution_plan_path), flush=True)

    if args.show:
        cv2.imshow("snapshot detector 3d", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
