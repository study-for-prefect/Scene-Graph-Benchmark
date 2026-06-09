#!/usr/bin/env python3
"""Check whether base-frame perception stays stable across camera poses."""

import argparse
import json
import math
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tabletop/object geometry from multiple snapshots.")
    parser.add_argument("snapshot_dirs", nargs="+", help="Snapshot output directories to compare.")
    parser.add_argument("--object-label", default="rectangle")
    parser.add_argument("--max-object-spread-mm", type=float, default=10.0)
    parser.add_argument("--max-table-height-spread-mm", type=float, default=10.0)
    parser.add_argument("--max-table-tilt-deg", type=float, default=5.0)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_object(state, requested_label):
    requested = requested_label.lower()
    matches = [obj for obj in state.get("objects", []) if requested in str(obj.get("label", "")).lower()]
    if len(matches) != 1:
        raise RuntimeError(
            "{}: expected exactly one object matching {!r}, found {}".format(
                state.get("frame_id", "snapshot"), requested_label, len(matches)
            )
        )
    obj = matches[0]
    center = obj.get("geometry_center_m") or obj.get("center_3d_base_m")
    if not center:
        raise RuntimeError("Object {!r} has no valid base-frame center.".format(obj.get("label")))
    return obj, np.asarray(center, dtype=float)


def table_tilt_deg(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))


def main():
    args = parse_args()
    if len(args.snapshot_dirs) < 2:
        raise RuntimeError("Provide at least two snapshot directories.")

    records = []
    for directory in args.snapshot_dirs:
        state = load_json(os.path.join(directory, "private_scene_state.json"))
        tabletop = load_json(os.path.join(directory, "tabletop_geometry.json"))
        table = tabletop.get("table_plane")
        if tabletop.get("status") != "ok" or not table:
            raise RuntimeError("{} has no valid table plane.".format(directory))
        obj, center = find_object(state, args.object_label)
        records.append(
            {
                "directory": directory,
                "object_label": obj.get("label"),
                "center": center,
                "table_origin": np.asarray(table["origin_m"], dtype=float),
                "table_normal": np.asarray(table["normal"], dtype=float),
                "table_tilt_deg": table_tilt_deg(table["normal"]),
            }
        )

    first = records[0]
    print("reference: {}".format(first["directory"]))
    for index, record in enumerate(records, 1):
        delta = (record["center"] - first["center"]) * 1000.0
        print(
            "pose {} object={} center={} delta_from_pose1_mm=[{:.1f}, {:.1f}, {:.1f}] "
            "table_z={:.4f} table_tilt_deg={:.2f}".format(
                index,
                record["object_label"],
                [round(float(v), 4) for v in record["center"]],
                delta[0],
                delta[1],
                delta[2],
                record["table_origin"][2],
                record["table_tilt_deg"],
            )
        )

    centers = np.vstack([record["center"] for record in records])
    center_mean = np.mean(centers, axis=0)
    object_spread_mm = float(np.max(np.linalg.norm(centers - center_mean, axis=1)) * 1000.0)
    table_heights = np.asarray([record["table_origin"][2] for record in records])
    table_height_spread_mm = float((np.max(table_heights) - np.min(table_heights)) * 1000.0)
    max_table_tilt_deg = float(max(record["table_tilt_deg"] for record in records))
    passed = (
        object_spread_mm <= args.max_object_spread_mm
        and table_height_spread_mm <= args.max_table_height_spread_mm
        and max_table_tilt_deg <= args.max_table_tilt_deg
    )
    print(
        "summary object_spread_mm={:.1f} table_height_spread_mm={:.1f} "
        "max_table_tilt_deg={:.2f} stability_pass={}".format(
            object_spread_mm,
            table_height_spread_mm,
            max_table_tilt_deg,
            passed,
        )
    )
    if not passed:
        print("Diagnosis: base<-camera hand-eye/TF is inconsistent across observation poses.")
        return 2
    print("Diagnosis: hand-eye/TF stability passed; investigate a remaining fixed base-frame bias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
