"""Bridge LLM decisions to ROS2 robot control topics.

This is intentionally conservative:
- preview mode is the default and publishes nothing;
- real motion requires --execute;
- only named joint poses and gripper open/close are executable here;
- object-coordinate actions are refused until camera->base transforms and
  motion planning are connected.
"""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_DECISION = "/tmp/scene_graph_closed_loop/scene_reasoning_decision.json"

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_JOINT_NAMES = [
    "gripper_finger1_joint",
    "gripper_finger2_joint",
]

NAMED_POSES = {
    "home": [0.0, -0.262, 1.047, 0.0, -1.57, 0.0],
    "ready": [0.0, -0.262, 1.047, 0.0, -1.57, 0.0],
    "observe": [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
}

GRIPPER_POSES = {
    "open_gripper": [0.0, 0.0],
    "close_gripper": [0.025, 0.025],
}

PROFILES = {
    "real_ur5": {
        "arm_topic": "/scaled_joint_trajectory_controller/joint_trajectory",
        "gripper_topic": "/ur5_gripper_controller/joint_trajectory",
    },
    "gazebo_custom": {
        "arm_topic": "/ur5_pos_joint_traj_controller/joint_trajectory",
        "gripper_topic": "/ur5_gripper_controller/joint_trajectory",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Send constrained LLM decisions to ROS2 robot topics.")
    parser.add_argument("--decision-json", default=DEFAULT_DECISION)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="real_ur5")
    parser.add_argument("--arm-topic", default="", help="Override arm JointTrajectory topic.")
    parser.add_argument("--gripper-topic", default="", help="Override gripper JointTrajectory topic.")
    parser.add_argument("--execute", action="store_true", help="Actually publish ROS2 trajectory messages.")
    parser.add_argument("--wait-subscribers", type=float, default=2.0)
    parser.add_argument("--motion-duration", type=float, default=3.0)
    parser.add_argument("--allow-unknown-controller", action="store_true")
    return parser.parse_args()


def load_json_or_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def action_steps(decision):
    if isinstance(decision.get("action_plan"), list):
        return decision["action_plan"]
    if isinstance(decision.get("steps"), list):
        return decision["steps"]
    if isinstance(decision.get("task_decision"), dict):
        step = dict(decision["task_decision"])
        step.setdefault("step", 1)
        return [step]
    if decision.get("action"):
        return [decision]
    raise ValueError("Decision JSON must contain action_plan, steps, task_decision, or action.")


def trajectory_msg(joint_names, positions, duration_sec):
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in positions]
    point.time_from_start.sec = int(duration_sec)
    point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
    msg.points = [point]
    return msg


class RobotBridge(Node):
    def __init__(self, arm_topic, gripper_topic):
        super().__init__("llm_robot_bridge")
        self.arm_pub = self.create_publisher(JointTrajectory, arm_topic, 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, gripper_topic, 10)
        self.arm_topic = arm_topic
        self.gripper_topic = gripper_topic

    def wait_for_subscribers(self, timeout_sec):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.arm_pub.get_subscription_count() > 0:
                return True
        return self.arm_pub.get_subscription_count() > 0

    def publish_arm_pose(self, pose_name, positions, duration_sec):
        msg = trajectory_msg(UR5_JOINT_NAMES, positions, duration_sec)
        self.arm_pub.publish(msg)
        self.get_logger().info("Published named pose '%s' to %s" % (pose_name, self.arm_topic))

    def publish_gripper(self, action, positions, duration_sec):
        msg = trajectory_msg(GRIPPER_JOINT_NAMES, positions, duration_sec)
        self.gripper_pub.publish(msg)
        self.get_logger().info("Published gripper action '%s' to %s" % (action, self.gripper_topic))


def compile_supported_command(step):
    action = step.get("action")
    if action == "move_named_pose":
        pose_name = step.get("named_pose") or step.get("pose") or step.get("target")
        if pose_name not in NAMED_POSES:
            return None, "unknown_named_pose"
        return {"type": "arm_pose", "name": pose_name, "positions": NAMED_POSES[pose_name]}, None

    if action in GRIPPER_POSES:
        return {"type": "gripper", "name": action, "positions": GRIPPER_POSES[action]}, None

    if action in ("ask_user", "stop"):
        return {"type": "no_motion", "name": action}, None

    if action in ("pick", "place_relative", "move_above"):
        return None, "coordinate_motion_not_enabled"

    return None, "unsupported_action"


def preview_commands(steps):
    compiled = []
    for step in steps:
        command, error = compile_supported_command(step)
        compiled.append(
            {
                "step": step.get("step"),
                "action": step.get("action"),
                "reason": step.get("reason", ""),
                "command": command,
                "error": error,
            }
        )
    return compiled


def main():
    args = parse_args()
    profile = PROFILES[args.profile]
    arm_topic = args.arm_topic or profile["arm_topic"]
    gripper_topic = args.gripper_topic or profile["gripper_topic"]

    decision = load_json_or_text(args.decision_json)
    steps = action_steps(decision)
    compiled = preview_commands(steps)

    print(json.dumps({"mode": "execute" if args.execute else "preview", "commands": compiled}, ensure_ascii=False, indent=2))

    if not args.execute:
        print("Preview only. Add --execute to publish ROS2 trajectory messages.")
        return

    rclpy.init()
    node = RobotBridge(arm_topic, gripper_topic)
    try:
        has_arm_subscriber = node.wait_for_subscribers(args.wait_subscribers)
        if not has_arm_subscriber and not args.allow_unknown_controller:
            raise RuntimeError(
                "No subscriber on arm topic %s. Start the UR driver/controller first, or pass --allow-unknown-controller."
                % arm_topic
            )

        for item in compiled:
            command = item["command"]
            if command is None:
                node.get_logger().warn(
                    "Skipping step %s action=%s error=%s"
                    % (item.get("step"), item.get("action"), item.get("error"))
                )
                continue
            if command["type"] == "arm_pose":
                node.publish_arm_pose(command["name"], command["positions"], args.motion_duration)
            elif command["type"] == "gripper":
                node.publish_gripper(command["name"], command["positions"], args.motion_duration)
            elif command["type"] == "no_motion":
                node.get_logger().info("No-motion action: %s" % command["name"])
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("robot bridge failed: %s" % exc, file=sys.stderr)
        raise
