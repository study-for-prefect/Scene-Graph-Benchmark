#!/usr/bin/env python3
"""Calibrate a tool-center-point translation with the pivot method.

Keep a rigid calibration tip at one fixed point while changing tool0
orientation. For every sample:

    base_pivot = base_R_tool0 * tool0_p_tcp + base_p_tool0

The least-squares solution yields tool0_p_tcp and the fixed pivot position.
This calibrates TCP translation only; TCP orientation must be defined from the
tool/gripper geometry.
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_OUTPUT = "/tmp/tcp_pivot_calibration.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate tool0->TCP translation by pivoting around a fixed point.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--tf-timeout", type=float, default=3.0)
    parser.add_argument("--input-json", default="", help="Solve previously captured poses instead of collecting TF.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rms-mm", type=float, default=2.0)
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument(
        "--min-rotation-deg",
        type=float,
        default=5.0,
        help="Reject consecutive samples whose tool orientation changes by less than this angle.",
    )
    return parser.parse_args()


def matrix_from_transform(transform):
    from robot_scene_pipeline.tf_transform import transform_to_matrix

    return transform_to_matrix(transform)


def rotation_difference_deg(first, second):
    relative = np.asarray(first, dtype=float).T.dot(np.asarray(second, dtype=float))
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def quaternion_to_matrix_xyzw(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def tool0_position_for_tcp(tcp_position_base, tool_quaternion_xyzw, tcp_offset_tool):
    tcp_position_base = np.asarray(tcp_position_base, dtype=float)
    tcp_offset_tool = np.asarray(tcp_offset_tool, dtype=float)
    rotation = quaternion_to_matrix_xyzw(tool_quaternion_xyzw)
    return (tcp_position_base - rotation.dot(tcp_offset_tool)).tolist()


def normalize_pose_matrix(item):
    if "matrix_4x4" in item:
        matrix = np.asarray(item["matrix_4x4"], dtype=float)
    elif "matrix" in item:
        matrix = np.asarray(item["matrix"], dtype=float)
    elif "rotation_3x3" in item and "translation_m" in item:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.asarray(item["rotation_3x3"], dtype=float)
        matrix[:3, 3] = np.asarray(item["translation_m"], dtype=float)
    else:
        raise ValueError("Each sample must contain matrix_4x4 or rotation_3x3 + translation_m.")
    if matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4, got {}.".format(matrix.shape))
    return matrix


def solve_pivot_calibration(pose_matrices):
    if len(pose_matrices) < 4:
        raise ValueError("Pivot calibration needs at least 4 poses; 8-12 diverse poses are recommended.")

    rows = []
    values = []
    for matrix in pose_matrices:
        matrix = np.asarray(matrix, dtype=float)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        rows.append(np.hstack([rotation, -np.eye(3, dtype=float)]))
        values.append(-translation)

    design = np.vstack(rows)
    target = np.concatenate(values)
    solution, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if rank < 6:
        raise ValueError(
            "Pose set is degenerate (rank {}). Use more varied tool orientations while keeping the tip fixed.".format(
                rank
            )
        )

    tcp_tool = solution[:3]
    pivot_base = solution[3:]
    errors = []
    for matrix in pose_matrices:
        predicted = matrix[:3, :3].dot(tcp_tool) + matrix[:3, 3]
        errors.append(predicted - pivot_base)
    errors = np.asarray(errors, dtype=float)
    error_norms = np.linalg.norm(errors, axis=1)
    rms_m = float(np.sqrt(np.mean(error_norms ** 2)))
    max_m = float(np.max(error_norms))
    condition = float(singular_values[0] / singular_values[-1])
    return {
        "tcp_offset_tool_m": tcp_tool.tolist(),
        "pivot_point_base_m": pivot_base.tolist(),
        "rms_error_m": rms_m,
        "max_error_m": max_m,
        "condition_number": condition,
        "rank": int(rank),
        "sample_errors_m": errors.tolist(),
    }


class TfCollector:
    def __init__(self, base_frame, tool_frame):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener

        self.rclpy = rclpy
        self.base_frame = base_frame
        self.tool_frame = tool_frame
        rclpy.init(args=None)
        self.node = rclpy.create_node("tcp_pivot_calibration")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)  # noqa F841
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def lookup(self, timeout_sec):
        from rclpy.duration import Duration
        from rclpy.time import Time

        deadline = time.time() + float(timeout_sec)
        last_error = None
        while time.time() < deadline:
            try:
                return self.buffer.lookup_transform(
                    self.base_frame,
                    self.tool_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        raise RuntimeError(
            "Failed to lookup TF {} <- {} within {:.2f}s: {}".format(
                self.base_frame, self.tool_frame, timeout_sec, last_error
            )
        )

    def close(self):
        self.executor.shutdown()
        self.spin_thread.join(timeout=1.0)
        self.node.destroy_node()
        self.rclpy.shutdown()


def collect_pose_matrices(args):
    collector = TfCollector(args.base_frame, args.tool_frame)
    poses = []
    try:
        print("Attach a rigid pointer whose tip represents the desired TCP.", flush=True)
        print("Keep the tip on one fixed dimple and vary tool orientation substantially.", flush=True)
        while len(poses) < args.sample_count:
            input("Pose {}/{} ready; press Enter to capture... ".format(len(poses) + 1, args.sample_count))
            transform = collector.lookup(args.tf_timeout)
            matrix = matrix_from_transform(transform)
            if poses:
                translation_change_mm = float(
                    np.linalg.norm(matrix[:3, 3] - poses[-1][:3, 3]) * 1000.0
                )
                rotation_change_deg = rotation_difference_deg(
                    poses[-1][:3, :3], matrix[:3, :3]
                )
            else:
                translation_change_mm = 0.0
                rotation_change_deg = 0.0
            if poses and rotation_change_deg < args.min_rotation_deg:
                print(
                    "REJECTED: orientation changed only {:.2f} deg (< {:.2f} deg). Move to a different pose and retry.".format(
                        rotation_change_deg, args.min_rotation_deg
                    ),
                    flush=True,
                )
                continue
            poses.append(matrix)
            print(
                "captured {} t=[{:.5f}, {:.5f}, {:.5f}] delta_t={:.2f} mm delta_rotation={:.2f} deg".format(
                    len(poses),
                    matrix[0, 3],
                    matrix[1, 3],
                    matrix[2, 3],
                    translation_change_mm,
                    rotation_change_deg,
                ),
                flush=True,
            )
    finally:
        collector.close()
    return poses


def load_pose_matrices(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", payload.get("poses", []))
    if not samples:
        raise ValueError("Input JSON contains no samples/poses.")
    return [normalize_pose_matrix(item) for item in samples]


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    args = parse_args()
    poses = load_pose_matrices(args.input_json) if args.input_json else collect_pose_matrices(args)
    result = solve_pivot_calibration(poses)
    result.update(
        {
            "schema_version": "tcp_pivot_calibration_v1",
            "timestamp": time.time(),
            "base_frame": args.base_frame,
            "tool_frame": args.tool_frame,
            "tcp_frame": "calibrated_tcp",
            "orientation_note": "Pivot calibration determines translation only; define TCP orientation separately.",
            "sample_count": len(poses),
            "samples": [{"matrix_4x4": matrix.tolist()} for matrix in poses],
            "quality_pass": result["rms_error_m"] <= args.max_rms_mm / 1000.0
            and result["condition_number"] <= args.max_condition_number,
            "max_rms_mm": args.max_rms_mm,
            "max_condition_number": args.max_condition_number,
        }
    )
    write_json(args.output, result)
    print("Saved TCP calibration: {}".format(args.output))
    print("tool0->tcp translation m: {}".format([round(v, 6) for v in result["tcp_offset_tool_m"]]))
    print(
        "RMS={:.3f} mm, max={:.3f} mm, condition={:.2f}, quality_pass={}".format(
            result["rms_error_m"] * 1000.0,
            result["max_error_m"] * 1000.0,
            result["condition_number"],
            result["quality_pass"],
        )
    )
    if not result["quality_pass"]:
        print("Calibration quality failed. Recollect with a rigid tip and more diverse orientations.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
