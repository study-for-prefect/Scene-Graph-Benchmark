#!/usr/bin/env python3
"""Classify grasp-center offsets as base-fixed, tool-rotating, or mixed."""

import argparse
import json
import math

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze measured XY grasp-center errors across tool yaw angles.")
    parser.add_argument(
        "--sample",
        action="append",
        nargs=3,
        type=float,
        metavar=("YAW_DEG", "ERROR_BASE_X_MM", "ERROR_BASE_Y_MM"),
        help="Measured gripper-center minus target-center error in base_link XY. Repeat at least 3 times.",
    )
    parser.add_argument(
        "--camera-sample",
        action="append",
        nargs=3,
        type=float,
        metavar=("YAW_DEG", "ERROR_CAMERA_X_MM", "ERROR_CAMERA_Y_MM"),
        help="Measured gripper-center minus target-center error in camera optical XY.",
    )
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def camera_error_to_base_xy(error_camera_xy_mm, matrix):
    error_optical = np.array(
        [float(error_camera_xy_mm[0]), float(error_camera_xy_mm[1]), 0.0],
        dtype=float,
    )
    error_camera_link = np.array([error_optical[2], -error_optical[0], -error_optical[1]], dtype=float)
    return np.asarray(matrix, dtype=float)[:3, :3].dot(error_camera_link)[:2]


def load_samples(args):
    samples = []
    for yaw, error_x, error_y in args.sample or []:
        samples.append([float(yaw), float(error_x), float(error_y)])
    if args.camera_sample:
        with open(args.tf_json, "r", encoding="utf-8") as f:
            matrix = np.asarray(json.load(f)["matrix_4x4"], dtype=float)
        for yaw, error_x, error_y in args.camera_sample:
            base_xy = camera_error_to_base_xy([error_x, error_y], matrix)
            samples.append([float(yaw), float(base_xy[0]), float(base_xy[1])])
    if len(samples) < 3:
        raise ValueError("Provide at least 3 --sample or --camera-sample measurements.")
    return np.asarray(samples, dtype=float)


def rotation_2d(yaw_deg):
    angle = math.radians(float(yaw_deg))
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=float)


def fit_offsets(samples):
    yaws = samples[:, 0]
    errors = samples[:, 1:3]

    base_offset = np.mean(errors, axis=0)
    base_predictions = np.tile(base_offset, (len(errors), 1))
    base_residuals = errors - base_predictions

    tool_errors = np.vstack([rotation_2d(-yaw).dot(error) for yaw, error in zip(yaws, errors)])
    tool_offset = np.mean(tool_errors, axis=0)
    tool_predictions = np.vstack([rotation_2d(yaw).dot(tool_offset) for yaw in yaws])
    tool_residuals = errors - tool_predictions

    rows = []
    values = []
    for yaw, error in zip(yaws, errors):
        rotation = rotation_2d(yaw)
        rows.append(np.hstack([np.eye(2), rotation]))
        values.append(error)
    mixed_solution, _, _, _ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    mixed_base = mixed_solution[:2]
    mixed_tool = mixed_solution[2:]
    mixed_predictions = np.vstack(
        [mixed_base + rotation_2d(yaw).dot(mixed_tool) for yaw in yaws]
    )
    mixed_residuals = errors - mixed_predictions

    def rms(residuals):
        return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    rms_values = {
        "base_fixed": rms(base_residuals),
        "tool_rotating": rms(tool_residuals),
        "mixed": rms(mixed_residuals),
    }
    best_simple = min(("base_fixed", "tool_rotating"), key=lambda name: rms_values[name])
    if rms_values["mixed"] < 0.65 * rms_values[best_simple]:
        classification = "mixed"
    else:
        classification = best_simple
    return {
        "classification": classification,
        "sample_count": int(len(samples)),
        "samples_base_xy_mm": samples.tolist(),
        "base_fixed_offset_mm": base_offset.tolist(),
        "tool_rotating_offset_mm": tool_offset.tolist(),
        "mixed_base_offset_mm": mixed_base.tolist(),
        "mixed_tool_offset_mm": mixed_tool.tolist(),
        "rms_error_mm": rms_values,
        "interpretation": {
            "base_fixed": "Use base-frame compensation; also verify hand-eye translation.",
            "tool_rotating": "The offset rotates with gripper yaw; measure or calibrate tool0->TCP.",
            "mixed": "Both base/hand-eye and tool/TCP offsets are present.",
        }[classification],
    }


def main():
    args = parse_args()
    result = fit_offsets(load_samples(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Saved analysis: {}".format(args.output))


if __name__ == "__main__":
    main()
