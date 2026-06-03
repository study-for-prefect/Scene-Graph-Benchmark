#!/usr/bin/env python3
"""Safely preview MoveIt motions from robot_execution_plan.json.

Default behavior is planning only. It does not descend to target points and
does not control the gripper. Use --execute to move only to approach points.
Run with ROS2 Python 3.10 after ur_robot_driver and ur_moveit are running.
"""

import argparse
import json
import sys
import time

import rclpy
from geometry_msgs.msg import Pose, Point
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


DEFAULT_PLAN = "/tmp/robot_scene_pipeline/robot_execution_plan.json"

# Verified downward TCP/tool0 orientation copied from the existing yolo_grasp
# MoveIt control script in this workspace.
DEFAULT_QUAT_XYZW = [0.9999, -0.0121, -0.0043, 0.0100]


def parse_args():
    parser = argparse.ArgumentParser(description="Plan or execute approach-only MoveIt motions.")
    parser.add_argument("--plan-json", default=DEFAULT_PLAN)
    parser.add_argument("--step", type=int, default=1, help="Plan one step number. Ignored by --all-approaches.")
    parser.add_argument("--all-approaches", action="store_true", help="Plan all steps that have approach_position_m.")
    parser.add_argument("--execute", action="store_true", help="Actually execute approach-only motion.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for interactive confirmation before execution.")
    parser.add_argument("--cartesian", dest="cartesian", action="store_true", default=True, help="Use Cartesian planning. This is the default.")
    parser.add_argument("--joint-space", dest="cartesian", action="store_false", help="Use normal MoveIt joint-space pose planning.")
    parser.add_argument("--cartesian-max-step", type=float, default=0.005)
    parser.add_argument("--cartesian-fraction-threshold", type=float, default=0.90)
    parser.add_argument("--tool-z-offset", type=float, default=0.15, help="Add this to plan z for tool0/flange pose.")
    parser.add_argument(
        "--tool-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Additional base_link offset added to tool0 goal after z offset. Use this to compensate gripper/TCP XY offset.",
    )
    parser.add_argument("--min-z", type=float, default=0.05, help="Reject tool0 goals below this base_link z.")
    parser.add_argument("--max-z", type=float, default=0.80, help="Reject tool0 goals above this base_link z.")
    parser.add_argument("--max-radius", type=float, default=0.90, help="Reject xy radius beyond this value.")
    parser.add_argument("--quat-xyzw", nargs=4, type=float, default=DEFAULT_QUAT_XYZW)
    parser.add_argument(
        "--orientation-mode",
        choices=("current", "fixed"),
        default="current",
        help="Use current base->tool0 orientation or fixed --quat-xyzw.",
    )
    parser.add_argument("--tf-timeout", type=float, default=3.0)
    parser.add_argument("--group-name", default=ur.MOVE_GROUP_ARM)
    parser.add_argument("--base-link", default=ur.base_link_name())
    parser.add_argument("--end-effector", default=ur.end_effector_name())
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--acceleration", type=float, default=0.2)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--max-joint-delta", type=float, default=1.2, help="Warn if any joint changes more than this radian value.")
    return parser.parse_args()


def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def selected_steps(plan, args):
    steps = [step for step in plan.get("steps", []) if step.get("status") == "planned"]
    steps = [step for step in steps if step.get("approach_position_m")]
    if args.all_approaches:
        return steps
    for step in steps:
        if int(step.get("step", -1)) == args.step:
            return [step]
    raise RuntimeError("No planned step {} with approach_position_m.".format(args.step))


def tool0_goal_from_approach(approach_position_m, tool_z_offset, tool_offset_base):
    x, y, z = [float(v) for v in approach_position_m]
    return [
        x + float(tool_offset_base[0]),
        y + float(tool_offset_base[1]),
        z + float(tool_z_offset) + float(tool_offset_base[2]),
    ]


def validate_goal(goal, args):
    x, y, z = goal
    radius = (x * x + y * y) ** 0.5
    if z < args.min_z:
        raise RuntimeError("Rejected goal z={:.3f} below min-z={:.3f}".format(z, args.min_z))
    if z > args.max_z:
        raise RuntimeError("Rejected goal z={:.3f} above max-z={:.3f}".format(z, args.max_z))
    if radius > args.max_radius:
        raise RuntimeError("Rejected goal xy radius={:.3f} above max-radius={:.3f}".format(radius, args.max_radius))


def xyz_delta(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def make_pose(position, quat_xyzw):
    pose = Pose()
    pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def transform_position_quat(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    return [float(t.x), float(t.y), float(t.z)], [float(q.x), float(q.y), float(q.z), float(q.w)]


def trajectory_points(trajectory):
    if trajectory is None:
        return None
    joint_traj = trajectory.joint_trajectory if hasattr(trajectory, "joint_trajectory") else trajectory
    return joint_traj.joint_names, joint_traj.points


def print_trajectory_summary(node, trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        node.get_logger().error("No trajectory returned.")
        return
    joint_names, points = parsed
    node.get_logger().info("Trajectory points: {}".format(len(points)))
    if not points:
        return
    for idx in sorted(set([0, len(points) - 1])):
        point = points[idx]
        t = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
        values = ", ".join(
            "{}={:.4f}".format(name, pos)
            for name, pos in zip(joint_names, point.positions)
        )
        node.get_logger().info("  point {} t={:.3f}: {}".format(idx, t, values))


def max_joint_delta(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if len(points) < 2:
        return None
    start = points[0].positions
    end = points[-1].positions
    deltas = [abs(float(b) - float(a)) for a, b in zip(start, end)]
    if not deltas:
        return None
    idx = max(range(len(deltas)), key=lambda i: deltas[i])
    return joint_names[idx], deltas[idx]


class MoveItPreviewNode(Node):
    def __init__(self, args):
        super().__init__("moveit_plan_preview")
        self.joint_state_seen = False
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)  # noqa F841
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=args.base_link,
            end_effector_name=args.end_effector,
            group_name=args.group_name,
            use_move_group_action=True,
        )
        self.moveit2.max_velocity = args.velocity
        self.moveit2.max_acceleration = args.acceleration
        self.moveit2.allowed_planning_time = args.planning_time

    def joint_state_cb(self, _msg):
        self.joint_state_seen = True

    def wait_for_joint_state(self, timeout=5.0):
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state_seen:
                return True
        return False

    def current_tool_quat_xyzw(self, timeout=3.0):
        transform = self.current_tool_transform(timeout=timeout)
        q = transform.transform.rotation
        return [float(q.x), float(q.y), float(q.z), float(q.w)]

    def current_tool_transform(self, timeout=3.0):
        deadline = time.time() + timeout
        last_error = None
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                return self.tf_buffer.lookup_transform(
                    "base_link",
                    "tool0",
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Failed to read current base_link->tool0 orientation: {}".format(last_error))


def main():
    args = parse_args()
    plan = load_plan(args.plan_json)
    if plan.get("coordinate_convention", {}).get("frame") != "base_link":
        print("ERROR: execution plan is not in base_link frame.", file=sys.stderr)
        return 2

    steps = selected_steps(plan, args)
    print("Loaded {} approach step(s) from {}".format(len(steps), args.plan_json))
    print("Mode: {}".format("EXECUTE approach only" if args.execute else "PLAN ONLY"))
    print("Planner: {}".format("Cartesian" if args.cartesian else "Joint-space pose"))

    rclpy.init(args=None)
    node = MoveItPreviewNode(args)
    try:
        if not node.wait_for_joint_state(timeout=5.0):
            node.get_logger().error("No /joint_states received. Start UR driver and MoveIt first.")
            return 2

        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        node.get_logger().info(
            "Current tool0 position=[{:.4f}, {:.4f}, {:.4f}], quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                current_pos[0], current_pos[1], current_pos[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
            )
        )

        for step in steps:
            current_tool = node.current_tool_transform(timeout=args.tf_timeout)
            current_pos, current_quat = transform_position_quat(current_tool)
            if args.orientation_mode == "current":
                quat_xyzw = current_quat
                node.get_logger().info(
                    "Using current tool0 orientation quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                        quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
                    )
                )
            else:
                quat_xyzw = args.quat_xyzw
                node.get_logger().info(
                    "Using fixed orientation quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                        quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
                    )
                )
            approach = step["approach_position_m"]
            tool_goal = tool0_goal_from_approach(approach, args.tool_z_offset, args.tool_offset_base)
            validate_goal(tool_goal, args)
            delta_xyz = [tool_goal[i] - current_pos[i] for i in range(3)]
            node.get_logger().info(
                "Step {} {}: object={} ref={} approach={} -> tool0_goal={} delta_xyz={}".format(
                    step.get("step"),
                    step.get("action"),
                    step.get("object_label"),
                    step.get("reference_label"),
                    [round(v, 4) for v in approach],
                    [round(v, 4) for v in tool_goal],
                    [round(v, 4) for v in delta_xyz],
                )
            )

            pose = make_pose(tool_goal, quat_xyzw)
            trajectory = node.moveit2.plan(
                pose=pose,
                cartesian=args.cartesian,
                max_step=args.cartesian_max_step,
                cartesian_fraction_threshold=args.cartesian_fraction_threshold,
            )
            if trajectory is None:
                node.get_logger().error("MoveIt planning failed for step {}.".format(step.get("step")))
                continue
            print_trajectory_summary(node, trajectory)
            delta = max_joint_delta(trajectory)
            delta_too_large = False
            if delta:
                joint_name, value = delta
                if value > args.max_joint_delta:
                    delta_too_large = True
                    node.get_logger().warning(
                        "Large joint change detected: {} delta={:.3f} rad > {:.3f}. Do not execute until reviewed.".format(
                            joint_name,
                            value,
                            args.max_joint_delta,
                        )
                    )

            if args.execute:
                if delta_too_large:
                    node.get_logger().error("Execution refused because joint delta exceeds safety threshold.")
                    continue
                if not args.yes:
                    confirm = input("Execute approach-only motion for step {}? Type yes: ".format(step.get("step")))
                    if confirm.strip().lower() != "yes":
                        node.get_logger().info("Execution skipped by user.")
                        continue
                node.moveit2.execute(trajectory)
                ok = node.moveit2.wait_until_executed()
                node.get_logger().info("Execution result: {}".format(ok))
                final_tool = node.current_tool_transform(timeout=args.tf_timeout)
                final_t = final_tool.transform.translation
                final_pos = [float(final_t.x), float(final_t.y), float(final_t.z)]
                err = xyz_delta(final_pos, tool_goal)
                node.get_logger().info(
                    "Final tool0 position={}, goal={}, error={}".format(
                        [round(v, 4) for v in final_pos],
                        [round(v, 4) for v in tool_goal],
                        [round(v, 4) for v in err],
                    )
                )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
