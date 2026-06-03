"""Generate hand-eye TF candidates as JSON.

Run this with ROS2 Python 3.10. It listens to base<-wrist TF and combines it
with the provided hand-eye calibration in several common conventions:

- xyzw wrist->camera
- xyzw camera->wrist, inverted before use
- wxyz wrist->camera
- wxyz camera->wrist, inverted before use

The detector can read the generated JSON in Python 3.7 without importing rclpy.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


DEFAULT_OUTPUT = "/tmp/scene_tf_handeye_candidates.json"
DEFAULT_TRANSLATION = [0.08599923, -0.03025062, 0.00174066]
DEFAULT_QUATERNION = [-0.706447, -0.01772374, -0.70735528, 0.01634021]


def parse_args():
    parser = argparse.ArgumentParser(description="Write hand-eye candidate base<-camera matrices to JSON.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--wrist-frame", default="wrist_3_link")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--translation", nargs=3, type=float, default=DEFAULT_TRANSLATION)
    parser.add_argument("--quaternion", nargs=4, type=float, default=DEFAULT_QUATERNION)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def quaternion_to_matrix_xyzw(quat):
    x, y, z, w = [float(v) for v in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_from_translation_quat(translation, quat_xyzw):
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = quaternion_to_matrix_xyzw(quat_xyzw)
    mat[:3, 3] = [float(v) for v in translation]
    return mat


def transform_to_matrix(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    return matrix_from_translation_quat([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def matrix_payload(name, description, matrix):
    return {
        "name": name,
        "description": description,
        "matrix_4x4": matrix.tolist(),
    }


def build_candidates(base_t_wrist, translation, quaternion):
    quat_xyzw = list(quaternion)
    quat_wxyz_as_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]

    wrist_t_camera_xyzw = matrix_from_translation_quat(translation, quat_xyzw)
    wrist_t_camera_wxyz = matrix_from_translation_quat(translation, quat_wxyz_as_xyzw)

    return [
        matrix_payload(
            "xyzw_wrist_to_camera",
            "Input quaternion interpreted as qx qy qz qw; translation/rotation is wrist->camera.",
            base_t_wrist.dot(wrist_t_camera_xyzw),
        ),
        matrix_payload(
            "xyzw_camera_to_wrist_inverted",
            "Input quaternion interpreted as qx qy qz qw; calibration was camera->wrist, so it is inverted.",
            base_t_wrist.dot(np.linalg.inv(wrist_t_camera_xyzw)),
        ),
        matrix_payload(
            "wxyz_wrist_to_camera",
            "Input quaternion interpreted as qw qx qy qz; translation/rotation is wrist->camera.",
            base_t_wrist.dot(wrist_t_camera_wxyz),
        ),
        matrix_payload(
            "wxyz_camera_to_wrist_inverted",
            "Input quaternion interpreted as qw qx qy qz; calibration was camera->wrist, so it is inverted.",
            base_t_wrist.dot(np.linalg.inv(wrist_t_camera_wxyz)),
        ),
    ]


def write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = rclpy.create_node("tf_handeye_candidates_json")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa F841
    period = 1.0 / max(args.rate, 0.1)
    print(
        "Writing hand-eye candidates using {} <- {} to {}".format(
            args.base_frame, args.wrist_frame, args.output
        ),
        flush=True,
    )
    try:
        while rclpy.ok():
            deadline = time.time() + args.timeout
            last_error = None
            transform = None
            while time.time() < deadline and transform is None:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    transform = buffer.lookup_transform(
                        args.base_frame,
                        args.wrist_frame,
                        Time(),
                        timeout=Duration(seconds=0.1),
                    )
                except Exception as exc:
                    last_error = exc
            if transform is None:
                print(
                    "TF lookup failed: {} <- {}: {}".format(
                        args.base_frame, args.wrist_frame, last_error
                    ),
                    flush=True,
                )
            else:
                base_t_wrist = transform_to_matrix(transform)
                payload = {
                    "timestamp": time.time(),
                    "base_frame": args.base_frame,
                    "wrist_frame": args.wrist_frame,
                    "translation_input": list(args.translation),
                    "quaternion_input": list(args.quaternion),
                    "candidates": build_candidates(base_t_wrist, args.translation, args.quaternion),
                }
                write_json_atomic(args.output, payload)
                print(
                    "ok candidates={} wrist_t=[{:.4f},{:.4f},{:.4f}]".format(
                        len(payload["candidates"]),
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ),
                    flush=True,
                )
                if args.once:
                    break
            time.sleep(period)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
