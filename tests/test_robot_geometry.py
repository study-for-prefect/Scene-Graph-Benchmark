import math
from types import SimpleNamespace

import numpy as np

from robot_scene_pipeline.tabletop_geometry import (
    attach_tabletop_geometry,
    estimate_object_footprint,
    fit_plane_ransac,
    make_plane_basis,
)
from robot_scene_pipeline.grasp_orientation import downward_quaternion_for_yaw, object_yaw_orientation
from tools.decision_to_execution import compile_plan
from tools.analyze_grasp_offset import fit_offsets
from tools.build_geometry_pick_plan import find_object
from tools.tcp_pivot_calibration import solve_pivot_calibration, tool0_position_for_tcp
from tools.two_stage_visual_pick import (
    has_planned_motion,
    reject_additional_pick_steps,
    reject_incomplete_remaining_motion_steps,
    remaining_plan_after_step,
    select_llm_pick_step,
)


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew.dot(skew)


class MockDepthFrame:
    def __init__(self, values):
        self.values = values

    def get_width(self):
        return self.values.shape[1]

    def get_height(self):
        return self.values.shape[0]

    def get_distance(self, x, y):
        return self.values[y, x]


def test_pivot_calibration_recovers_tcp_offset():
    tcp = np.array([0.012, -0.018, 0.154])
    pivot = np.array([0.45, 0.10, 0.22])
    poses = []
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    for index, axis in enumerate(axes):
        rotation = axis_angle_matrix(axis, 0.25 + index * 0.19)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = pivot - rotation.dot(tcp)
        poses.append(matrix)

    result = solve_pivot_calibration(poses)

    np.testing.assert_allclose(result["tcp_offset_tool_m"], tcp, atol=1e-9)
    np.testing.assert_allclose(result["pivot_point_base_m"], pivot, atol=1e-9)
    assert result["rms_error_m"] < 1e-9


def test_tcp_target_is_converted_in_tool_orientation():
    tcp_target = [0.4, 0.1, 0.2]
    tcp_offset_tool = [0.0, 0.0, 0.15]
    downward_tool_quaternion = [1.0, 0.0, 0.0, 0.0]

    tool0 = tool0_position_for_tcp(tcp_target, downward_tool_quaternion, tcp_offset_tool)

    np.testing.assert_allclose(tool0, [0.4, 0.1, 0.35], atol=1e-9)


def test_plane_and_object_yaw_estimation():
    rng = np.random.RandomState(11)
    table_xy = rng.uniform([-0.4, -0.3], [0.4, 0.3], size=(4000, 2))
    table = np.column_stack([table_xy, rng.normal(0.0, 0.0008, size=len(table_xy))])
    outliers = rng.uniform([-0.4, -0.3, 0.03], [0.4, 0.3, 0.30], size=(300, 3))
    plane = fit_plane_ransac(
        np.vstack([table, outliers]),
        distance_threshold_m=0.004,
        iterations=300,
        min_inliers=1000,
        up_hint=[0, 0, 1],
        min_up_alignment=0.9,
    )
    plane["x_axis"], plane["y_axis"] = make_plane_basis(plane["normal"], [1, 0, 0])

    expected_yaw = math.radians(32.0)
    long_axis = np.array([math.cos(expected_yaw), math.sin(expected_yaw)])
    short_axis = np.array([-math.sin(expected_yaw), math.cos(expected_yaw)])
    long_values = rng.uniform(-0.10, 0.10, size=3000)
    short_values = rng.uniform(-0.035, 0.035, size=3000)
    xy = (
        np.array([0.12, -0.08])
        + long_values[:, None] * long_axis
        + short_values[:, None] * short_axis
    )
    object_points = np.column_stack([xy, rng.uniform(0.01, 0.08, size=len(xy))])

    geometry = estimate_object_footprint(object_points, plane, min_points=100)

    assert geometry["yaw_valid"]
    assert abs(geometry["yaw_deg"] - 32.0) < 2.0
    assert abs(geometry["dimensions_m"][0] - 0.20) < 0.015
    assert abs(geometry["dimensions_m"][1] - 0.07) < 0.015
    assert abs(geometry["dimensions_m"][2] - 0.08) < 0.01


def test_attach_tabletop_geometry_from_depth_frame():
    depth = np.ones((80, 100), dtype=float)
    depth[30:50, 35:65] = 0.90
    depth_frame = MockDepthFrame(depth)
    intrinsics = SimpleNamespace(fx=100.0, fy=100.0, ppx=50.0, ppy=40.0)
    detections = [
        {"id": 0, "label": "workspace", "confidence": 0.9, "bbox": [0, 0, 99, 79]},
        {"id": 1, "label": "block", "confidence": 0.9, "bbox": [35, 30, 64, 49]},
    ]
    args = SimpleNamespace(
        plane_point_stride=2,
        plane_max_depth_m=2.0,
        plane_distance_threshold_m=0.005,
        plane_ransac_iterations=200,
        plane_min_inliers=500,
        plane_min_up_alignment=0.8,
        object_point_stride=1,
        object_min_height_m=0.005,
        object_max_height_m=0.5,
        object_min_points=100,
        yaw_min_aspect_ratio=1.2,
        tf_point_mode="direct",
        base_frame="base_link",
    )
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    base_from_camera[2, 3] = 1.0

    output, table = attach_tabletop_geometry(
        detections,
        depth_frame,
        intrinsics,
        args,
        transform_matrix=base_from_camera,
    )

    assert table["frame"] == "base_link"
    assert table["normal"][2] > 0.99
    assert table["rms_error_m"] < 1e-6
    assert output[1]["pointcloud_geometry_valid"]
    assert output[1]["table_yaw_valid"]
    assert 0.08 < output[1]["dimensions_m"][2] < 0.12


def test_object_yaw_orientation_selects_nearest_180_degree_equivalent():
    base_quaternion = [0.0, 0.0, 0.0, 1.0]
    current_quaternion = downward_quaternion_for_yaw(base_quaternion, 205.0)

    quaternion, selected_yaw = object_yaw_orientation(
        28.0,
        True,
        "base_link",
        base_quaternion,
        current_quaternion,
        grasp_axis="long",
        yaw_sign="positive",
        yaw_offset_deg=0.0,
    )

    assert abs(selected_yaw - (-152.0)) < 1e-9
    np.testing.assert_allclose(quaternion, downward_quaternion_for_yaw(base_quaternion, 208.0), atol=1e-9)


def test_object_yaw_orientation_supports_negative_sign_and_short_axis():
    base_quaternion = [0.0, 0.0, 0.0, 1.0]

    _, selected_yaw = object_yaw_orientation(
        30.0,
        True,
        "base_link",
        base_quaternion,
        base_quaternion,
        grasp_axis="short",
        yaw_sign="negative",
        yaw_offset_deg=5.0,
    )

    assert abs(selected_yaw - 65.0) < 1e-9


def test_execution_plan_prefers_geometry_center_and_carries_yaw():
    state = {
        "frame_id": "test",
        "timestamp": 0.0,
        "base_frame": "base_link",
        "objects": [
            {
                "id": 1,
                "label": "rectangle",
                "center_3d_base_m": [0.5, 0.1, 0.01],
                "geometry_center_m": [0.51, 0.11, 0.02],
                "geometry_frame": "base_link",
                "dimensions_m": [0.06, 0.03, 0.02],
                "table_yaw_deg": 28.0,
                "table_yaw_valid": True,
            }
        ],
    }
    args = SimpleNamespace(
        place_offset_m=0.08,
        approach_height_m=0.10,
        pick_target_lift_m=0.0,
        place_target_lift_m=0.0,
        left_right_axis="y",
        left_direction_sign="positive",
        front_back_axis="x",
        front_direction_sign="positive",
    )

    plan = compile_plan({"action_plan": [{"step": 1, "action": "pick", "object_id": 1}]}, state, args)
    step = plan["steps"][0]

    assert step["target_position_m"] == [0.51, 0.11, 0.02]
    assert step["target_yaw_deg"] == 28.0
    assert step["target_yaw_valid"]


def test_two_stage_llm_pick_selection_and_remaining_plan():
    plan = {
        "execution_status": "not_executed",
        "steps": [
            {"step": 1, "action": "pick", "object_id": 3, "object_label": "square green", "status": "planned"},
            {
                "step": 2,
                "action": "place_relative",
                "object_id": 3,
                "status": "planned",
                "approach_position_m": [0.4, 0.1, 0.2],
            },
        ],
    }

    selected_index, selected = select_llm_pick_step(plan)
    remaining = remaining_plan_after_step(plan, selected_index)

    assert selected["object_id"] == 3
    assert [step["action"] for step in remaining["steps"]] == ["place_relative"]
    assert has_planned_motion(remaining)
    assert len(plan["steps"]) == 2


def test_second_snapshot_label_match_prefers_first_target_xy():
    state = {
        "objects": [
            {"id": 7, "label": "square green", "center_3d_base_m": [0.30, 0.10, 0.02]},
            {"id": 2, "label": "square green", "center_3d_base_m": [0.50, -0.10, 0.02]},
        ]
    }

    selected = find_object(state, object_label="square green", nearest_base_xy=[0.49, -0.11])

    assert selected["id"] == 2


def test_two_stage_rejects_a_second_llm_pick():
    plan = {
        "steps": [
            {"step": 1, "action": "pick", "status": "planned"},
            {"step": 2, "action": "pick", "status": "planned"},
        ]
    }

    try:
        reject_additional_pick_steps(plan, 0)
    except RuntimeError as exc:
        assert "one pick per run" in str(exc)
    else:
        raise AssertionError("Expected a second planned pick to be rejected.")


def test_two_stage_rejects_incomplete_post_pick_motion():
    plan = {
        "steps": [
            {"step": 1, "action": "pick", "status": "planned"},
            {"step": 2, "action": "place_relative", "status": "missing_coordinate_or_reference"},
        ]
    }

    try:
        reject_incomplete_remaining_motion_steps(plan, 0)
    except RuntimeError as exc:
        assert "not executable" in str(exc)
    else:
        raise AssertionError("Expected an incomplete post-pick motion to be rejected.")


def test_offset_analysis_distinguishes_base_and_tool_offsets():
    base_samples = np.array([[0.0, 5.0, -10.0], [45.0, 5.0, -10.0], [90.0, 5.0, -10.0]])
    base_result = fit_offsets(base_samples)
    assert base_result["classification"] == "base_fixed"

    tool_samples = np.array([[0.0, 10.0, 0.0], [90.0, 0.0, 10.0], [180.0, -10.0, 0.0]])
    tool_result = fit_offsets(tool_samples)
    assert tool_result["classification"] == "tool_rotating"
