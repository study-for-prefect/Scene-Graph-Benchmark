"""Realtime detector + RGB-D coordinate direction check.

This script is for hand-eye/TF sanity checks. It opens RealSense directly,
runs the custom detector, prints camera-frame 3D points, and optionally prints
base-frame points through TF.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_TF_JSON = "/tmp/scene_tf_base_camera.json"
DEFAULT_HAND_EYE_CANDIDATES_JSON = "/tmp/scene_tf_handeye_candidates.json"

from robot_scene_pipeline.depth_geometry import attach_3d, color_for_label  # noqa E402
from robot_scene_pipeline.detector_runtime import DEFAULT_CONFIG, DEFAULT_WEIGHT, DetectorModel  # noqa E402
from robot_scene_pipeline.tf_transform import apply_transform, transform_to_matrix  # noqa E402


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime custom detector coordinate direction check.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--detector-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--detect-every", type=int, default=5, help="Run detector every N frames.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Print coordinate table every N seconds.")
    parser.add_argument("--use-tf", action="store_true", help="Also transform camera points into base frame.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--tf-refresh-every", type=float, default=0.5)
    parser.add_argument(
        "--tf-point-mode",
        choices=("direct", "optical-to-camera-link"),
        default="direct",
        help="How to interpret pyrealsense2 optical xyz before applying base<-camera TF.",
    )
    parser.add_argument(
        "--tf-json",
        default=DEFAULT_TF_JSON,
        help="Read base<-camera matrix JSON written by tools/tf_lookup_json.py instead of importing rclpy.",
    )
    parser.add_argument(
        "--handeye-candidates-json",
        default="",
        help="Read candidate base<-camera matrices written by tools/tf_handeye_candidates_json.py.",
    )
    parser.add_argument(
        "--candidate-point-modes",
        choices=("direct", "optical-to-camera-link", "both"),
        default="both",
    )
    parser.add_argument(
        "--active-candidate",
        default="",
        help="Candidate key used for the overlay base[...] value. Empty means the first candidate.",
    )
    parser.add_argument(
        "--candidate-target-id",
        type=int,
        default=-1,
        help="Detection id to print candidate table for. -1 means first non-workspace object.",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


class RealtimeTfLookup:
    def __init__(self, base_frame, camera_frame):
        import rclpy
        from tf2_ros import Buffer, TransformListener

        self.rclpy = rclpy
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("realtime_detector_tf_check")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)  # noqa F841

    def lookup(self, timeout_sec):
        from rclpy.duration import Duration
        from rclpy.time import Time

        deadline = time.time() + timeout_sec
        last_error = None
        while time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            try:
                transform = self.buffer.lookup_transform(
                    self.base_frame,
                    self.camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                return transform_to_matrix(transform), transform
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Failed to lookup TF {} <- {} within {:.2f}s: {}".format(
                self.base_frame, self.camera_frame, timeout_sec, last_error
            )
        )

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def configure_streams(args):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    return rs, pipeline, align, intrinsics


def read_frame(pipeline, align, timeout_ms=5000):
    frames = pipeline.wait_for_frames(timeout_ms)
    aligned = align.process(frames)
    color = aligned.get_color_frame()
    depth = aligned.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), depth


def maybe_add_base_coordinates(detections, matrix):
    for det in detections:
        point = det.get("center_3d_m")
        if not point:
            det["center_3d_base_m"] = None
            det["base_coordinate_valid"] = False
            continue
        det["center_3d_base_m"] = apply_transform(matrix, point)
        det["base_coordinate_valid"] = True
    return detections


def point_for_tf(point, mode):
    if mode == "direct":
        return point
    if mode == "optical-to-camera-link":
        x_opt, y_opt, z_opt = point
        return [z_opt, -x_opt, -y_opt]
    raise ValueError("Unsupported tf point mode: {}".format(mode))


def maybe_add_base_coordinates_with_mode(detections, matrix, mode):
    for det in detections:
        point = det.get("center_3d_m")
        if not point:
            det["center_3d_base_m"] = None
            det["base_coordinate_valid"] = False
            continue
        det["center_3d_for_tf_m"] = point_for_tf(point, mode)
        det["center_3d_base_m"] = apply_transform(matrix, det["center_3d_for_tf_m"])
        det["base_coordinate_valid"] = True
    return detections


def load_tf_json_matrix(path, max_age_sec=2.0):
    if not os.path.exists(path):
        raise RuntimeError(
            "TF JSON file does not exist: {}. Start tools/tf_lookup_json.py first.".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    age = time.time() - float(payload.get("timestamp", 0.0))
    if age > max_age_sec:
        print("Warning: TF JSON is stale: {:.2f}s old".format(age), flush=True)
    return np.array(payload["matrix_4x4"], dtype=float), payload


def load_candidate_json(path, max_age_sec=2.0):
    if not os.path.exists(path):
        raise RuntimeError(
            "Candidate JSON file does not exist: {}. Start tools/tf_handeye_candidates_json.py first.".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    age = time.time() - float(payload.get("timestamp", 0.0))
    if age > max_age_sec:
        print("Warning: candidate JSON is stale: {:.2f}s old".format(age), flush=True)
    candidates = []
    for item in payload.get("candidates", []):
        candidates.append(
            {
                "name": item["name"],
                "description": item.get("description", ""),
                "matrix": np.array(item["matrix_4x4"], dtype=float),
            }
        )
    if not candidates:
        raise RuntimeError("Candidate JSON has no candidates: {}".format(path))
    return candidates, payload


def print_tf_json_summary(payload, path):
    print(
        "TF JSON OK: {} <- {}, file={}, t=[{:.4f},{:.4f},{:.4f}]".format(
            payload.get("parent_frame"),
            payload.get("child_frame"),
            path,
            *payload.get("translation", [0.0, 0.0, 0.0]),
        ),
        flush=True,
    )


def candidate_point_modes(mode):
    if mode == "both":
        return ["direct", "optical-to-camera-link"]
    return [mode]


def candidate_key(candidate_name, point_mode):
    return "{}|{}".format(candidate_name, point_mode)


def add_handeye_candidate_coordinates(detections, candidates, point_modes, active_key):
    first_key = None
    for det in detections:
        point = det.get("center_3d_m")
        det["candidate_base_m"] = {}
        if not point:
            det["center_3d_base_m"] = None
            det["base_coordinate_valid"] = False
            continue
        for candidate in candidates:
            for mode in point_modes:
                key = candidate_key(candidate["name"], mode)
                if first_key is None:
                    first_key = key
                point_tf = point_for_tf(point, mode)
                det["candidate_base_m"][key] = apply_transform(candidate["matrix"], point_tf)
        selected_key = active_key or first_key
        det["active_candidate_key"] = selected_key
        det["center_3d_base_m"] = det["candidate_base_m"].get(selected_key)
        det["base_coordinate_valid"] = det["center_3d_base_m"] is not None
    return detections


def choose_candidate_target(detections, target_id):
    if target_id >= 0:
        for det in detections:
            if det.get("id") == target_id:
                return det
        return None
    for det in detections:
        if det.get("label") != "workspace" and det.get("center_3d_m"):
            return det
    for det in detections:
        if det.get("center_3d_m"):
            return det
    return None


def draw_overlay(frame_bgr, detections, args):
    output = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = color_for_label(det["label_id"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        line_y = max(18, y1 - 28)
        text = "{}:{} {:.2f}".format(det["id"], det["label"], det["confidence"])
        cv2.putText(output, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        line_y += 16

        if det.get("center_3d_m"):
            x, y, z = det["center_3d_m"]
            cam_text = "cam[{:.2f},{:.2f},{:.2f}]".format(x, y, z)
        else:
            cam_text = "cam:null"
        cv2.putText(output, cam_text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        line_y += 16

        if args.use_tf:
            if det.get("center_3d_base_m"):
                bx, by, bz = det["center_3d_base_m"]
                base_text = "base[{:.2f},{:.2f},{:.2f}]".format(bx, by, bz)
            else:
                base_text = "base:null"
            cv2.putText(output, base_text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return output


def print_table(detections, use_tf):
    print("\n{} detections".format(len(detections)), flush=True)
    print("id label conf center_px camera_xyz_m{} depth".format(" base_xyz_m" if use_tf else ""), flush=True)
    for det in detections:
        cam = det.get("center_3d_m")
        cam_text = "[{:.4f},{:.4f},{:.4f}]".format(*cam) if cam else "null"
        if use_tf:
            base = det.get("center_3d_base_m")
            base_text = "[{:.4f},{:.4f},{:.4f}]".format(*base) if base else "null"
            print(
                "{} {} {:.3f} {} {} {} {:.4f}".format(
                    det["id"],
                    det["label"],
                    det["confidence"],
                    det.get("center_px"),
                    cam_text,
                    base_text,
                    det.get("depth_m", 0.0),
                ),
                flush=True,
            )
        else:
            print(
                "{} {} {:.3f} {} {} {:.4f}".format(
                    det["id"],
                    det["label"],
                    det["confidence"],
                    det.get("center_px"),
                    cam_text,
                    det.get("depth_m", 0.0),
                ),
                flush=True,
            )


def print_candidate_table(detections, target_id):
    target = choose_candidate_target(detections, target_id)
    if not target:
        print("No valid target for candidate table.", flush=True)
        return
    cam = target.get("center_3d_m")
    cam_text = "[{:.4f},{:.4f},{:.4f}]".format(*cam) if cam else "null"
    print(
        "\nCandidate base coordinates for id={} label={} cam={}".format(
            target["id"], target["label"], cam_text
        ),
        flush=True,
    )
    for key in sorted(target.get("candidate_base_m", {})):
        value = target["candidate_base_m"][key]
        print("  {}: [{:.4f},{:.4f},{:.4f}]".format(key, value[0], value[1], value[2]), flush=True)


def print_candidate_keys(candidates, point_modes):
    print("Candidate keys:", flush=True)
    for candidate in candidates:
        for mode in point_modes:
            print("  {}".format(candidate_key(candidate["name"], mode)), flush=True)


def main():
    args = parse_args()
    if not args.handeye_candidates_json and os.path.exists(DEFAULT_HAND_EYE_CANDIDATES_JSON):
        args.handeye_candidates_json = DEFAULT_HAND_EYE_CANDIDATES_JSON
        print(
            "Auto-enabled hand-eye candidate mode because candidate JSON exists: {}".format(
                args.handeye_candidates_json
            ),
            flush=True,
        )
    if args.handeye_candidates_json:
        args.use_tf = True
    elif not args.use_tf and args.tf_json and os.path.exists(args.tf_json):
        args.use_tf = True
        print("Auto-enabled TF because TF JSON exists: {}".format(args.tf_json), flush=True)
    print("Loading detector weight: {}".format(args.weight), flush=True)
    detector = DetectorModel(args)

    tf_matrix = None
    tf_lookup = None
    candidate_matrices = None
    candidate_modes = candidate_point_modes(args.candidate_point_modes)
    last_tf_refresh = 0.0
    if args.use_tf:
        if args.handeye_candidates_json:
            candidate_matrices, candidate_payload = load_candidate_json(
                args.handeye_candidates_json,
                max_age_sec=args.tf_timeout + 1.0,
            )
            print(
                "Hand-eye candidates OK: file={}, candidates={}, point_modes={}".format(
                    args.handeye_candidates_json,
                    len(candidate_matrices),
                    ",".join(candidate_modes),
                ),
                flush=True,
            )
            print_candidate_keys(candidate_matrices, candidate_modes)
            print(
                "base_frame={} wrist_frame={}".format(
                    candidate_payload.get("base_frame"), candidate_payload.get("wrist_frame")
                ),
                flush=True,
            )
        elif args.tf_json:
            tf_matrix, tf_payload = load_tf_json_matrix(args.tf_json, max_age_sec=args.tf_timeout + 1.0)
            print_tf_json_summary(tf_payload, args.tf_json)
        else:
            tf_lookup = RealtimeTfLookup(args.base_frame, args.camera_frame)
            tf_matrix, transform = tf_lookup.lookup(args.tf_timeout)
            t = transform.transform.translation
            q = transform.transform.rotation
            print(
                "TF OK: {} <- {}, t=[{:.4f},{:.4f},{:.4f}], q=[{:.4f},{:.4f},{:.4f},{:.4f}]".format(
                    args.base_frame,
                    args.camera_frame,
                    t.x,
                    t.y,
                    t.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                ),
                flush=True,
            )
        last_tf_refresh = time.time()

    _, pipeline, align, intrinsics = configure_streams(args)
    latest_detections = []
    last_print = 0.0
    frame_idx = 0
    try:
        for _ in range(max(0, args.warmup_frames)):
            read_frame(pipeline, align)

        print("Realtime check started. Press q in the image window to quit.", flush=True)
        print("Camera xyz convention: x right, y down, z forward.", flush=True)
        if args.use_tf:
            print("Base xyz should follow your robot base_link axes.", flush=True)
            if args.handeye_candidates_json:
                print("Hand-eye candidate mode: {}".format(args.handeye_candidates_json), flush=True)
            else:
                print("TF point mode: {}".format(args.tf_point_mode), flush=True)

        while True:
            frame_bgr, depth_frame = read_frame(pipeline, align)
            if frame_bgr is None:
                continue

            if frame_idx % max(1, args.detect_every) == 0:
                if args.use_tf and time.time() - last_tf_refresh >= args.tf_refresh_every:
                    if args.handeye_candidates_json:
                        candidate_matrices, _ = load_candidate_json(
                            args.handeye_candidates_json,
                            max_age_sec=args.tf_timeout + 1.0,
                        )
                    elif args.tf_json:
                        tf_matrix, _ = load_tf_json_matrix(args.tf_json, max_age_sec=args.tf_timeout + 1.0)
                    else:
                        tf_matrix, _ = tf_lookup.lookup(args.tf_timeout)
                    last_tf_refresh = time.time()
                detections, _ = detector.predict(
                    frame_bgr,
                    args.score_thresh,
                    args.detector_scale,
                    args.max_detections,
                )
                detections = attach_3d(detections, depth_frame, intrinsics, args.depth_window)
                if args.use_tf:
                    if candidate_matrices:
                        detections = add_handeye_candidate_coordinates(
                            detections,
                            candidate_matrices,
                            candidate_modes,
                            args.active_candidate,
                        )
                    else:
                        detections = maybe_add_base_coordinates_with_mode(
                            detections,
                            tf_matrix,
                            args.tf_point_mode,
                        )
                latest_detections = detections

            now = time.time()
            if now - last_print >= args.print_every:
                print_table(latest_detections, args.use_tf)
                if candidate_matrices:
                    print_candidate_table(latest_detections, args.candidate_target_id)
                last_print = now

            if not args.no_display:
                output = draw_overlay(frame_bgr, latest_detections, args)
                cv2.imshow("realtime detector tf check", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        pipeline.stop()
        if tf_lookup is not None:
            tf_lookup.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
