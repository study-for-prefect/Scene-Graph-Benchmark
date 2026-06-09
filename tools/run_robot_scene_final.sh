#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 0 ]]; then
  INSTRUCTION="$*"
elif [[ -z "${INSTRUCTION:-}" ]]; then
  read -r -p "请输入操作指令: " INSTRUCTION
fi

if [[ -z "${INSTRUCTION:-}" ]]; then
  echo "ERROR: instruction is empty." >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/robot_scene_pipeline}"
SECOND_OUTPUT_DIR="${SECOND_OUTPUT_DIR:-${OUTPUT_DIR}_second}"
TF_JSON="${TF_JSON:-/tmp/scene_tf_base_camera.json}"
CONDA_ENV="${CONDA_ENV:-scene_graph_benchmark}"
MODEL="${MODEL:-qwen2.5vl:7b-q4_K_M}"

INTEGRATED_TWO_STAGE_PICK="${INTEGRATED_TWO_STAGE_PICK:-1}"
RUN_SNAPSHOT="${RUN_SNAPSHOT:-1}"
EXECUTE="${EXECUTE:-1}"
PATH_MODE="${PATH_MODE:-full}"
ENABLE_GRIPPER="${ENABLE_GRIPPER:-1}"
AUTO_YES="${AUTO_YES:-1}"

TOOL_OFFSET_X="${TOOL_OFFSET_X:--0.015}"
TOOL_OFFSET_Y="${TOOL_OFFSET_Y:-0}"
TOOL_OFFSET_Z="${TOOL_OFFSET_Z:-0}"
VELOCITY="${VELOCITY:-0.05}"
ACCELERATION="${ACCELERATION:-0.05}"
PLACE_OFFSET_M="${PLACE_OFFSET_M:-0.08}"
APPROACH_HEIGHT_M="${APPROACH_HEIGHT_M:-0.05}"
PICK_TARGET_LIFT_M="${PICK_TARGET_LIFT_M:-0.005}"
PLACE_TARGET_LIFT_M="${PLACE_TARGET_LIFT_M:-0.03}"

LEFT_RIGHT_AXIS="${LEFT_RIGHT_AXIS:-y}"
LEFT_DIRECTION_SIGN="${LEFT_DIRECTION_SIGN:-positive}"
FRONT_BACK_AXIS="${FRONT_BACK_AXIS:-x}"
FRONT_DIRECTION_SIGN="${FRONT_DIRECTION_SIGN:-positive}"

GRIPPER_PORT="${GRIPPER_PORT:-/dev/ttyUSB0}"
GRIPPER_FORCE="${GRIPPER_FORCE:-50}"
GRIPPER_SPEED="${GRIPPER_SPEED:-30}"
GRIPPER_OPEN_POSITION="${GRIPPER_OPEN_POSITION:-1000}"
GRIPPER_CLOSE_POSITION="${GRIPPER_CLOSE_POSITION:-0}"
READY_POSE_JSON="${READY_POSE_JSON:-config/rectangle_ready_pose.json}"
PRE_ROTATE_WRIST_DIRECTION="${PRE_ROTATE_WRIST_DIRECTION:-negative}"

PLAN_JSON="$OUTPUT_DIR/robot_execution_plan.json"

echo "Project: $PROJECT_ROOT"
echo "Instruction: $INSTRUCTION"
echo "Output dir: $OUTPUT_DIR"
echo "TF JSON: $TF_JSON"
echo "Tool offset base: [$TOOL_OFFSET_X, $TOOL_OFFSET_Y, $TOOL_OFFSET_Z]"
echo "Left/right: axis=$LEFT_RIGHT_AXIS, left_sign=$LEFT_DIRECTION_SIGN"
echo "Front/back: axis=$FRONT_BACK_AXIS, front_sign=$FRONT_DIRECTION_SIGN"
echo "Pick target lift: $PICK_TARGET_LIFT_M m"
echo "Place target lift: $PLACE_TARGET_LIFT_M m"

if [[ "$INTEGRATED_TWO_STAGE_PICK" == "1" ]]; then
  TWO_STAGE_CMD=(
    python3 tools/two_stage_visual_pick.py
    --instruction "$INSTRUCTION"
    --model "$MODEL"
    --first-dir "$OUTPUT_DIR"
    --second-dir "$SECOND_OUTPUT_DIR"
    --tf-json "$TF_JSON"
    --ready-pose-json "$READY_POSE_JSON"
    --conda-env "$CONDA_ENV"
    --tool-offset-base "$TOOL_OFFSET_X" "$TOOL_OFFSET_Y" "$TOOL_OFFSET_Z"
    --approach-height-m "$APPROACH_HEIGHT_M"
    --pick-target-lift-m "$PICK_TARGET_LIFT_M"
    --place-target-lift-m "$PLACE_TARGET_LIFT_M"
    --place-offset-m "$PLACE_OFFSET_M"
    --left-right-axis "$LEFT_RIGHT_AXIS"
    --left-direction-sign "$LEFT_DIRECTION_SIGN"
    --front-back-axis "$FRONT_BACK_AXIS"
    --front-direction-sign "$FRONT_DIRECTION_SIGN"
    --pre-rotate-wrist-direction "$PRE_ROTATE_WRIST_DIRECTION"
    --velocity "$VELOCITY"
    --acceleration "$ACCELERATION"
    --gripper-port "$GRIPPER_PORT"
  )
  if [[ "$ENABLE_GRIPPER" != "1" ]]; then
    TWO_STAGE_CMD+=(--disable-gripper)
  fi
  if [[ "$EXECUTE" == "1" ]]; then
    TWO_STAGE_CMD+=(--execute)
  fi
  if [[ "$AUTO_YES" == "1" ]]; then
    TWO_STAGE_CMD+=(--yes)
  fi

  echo
  echo "=== Integrated LLM + two-stage visual pick ==="
  echo "Second output dir: $SECOND_OUTPUT_DIR"
  echo "Command: ${TWO_STAGE_CMD[*]}"
  exec "${TWO_STAGE_CMD[@]}"
fi

if [[ ! -f "$TF_JSON" ]]; then
  echo "ERROR: TF JSON not found: $TF_JSON" >&2
  echo "Start tools/tf_lookup_json.py first, then rerun this script." >&2
  exit 2
fi

if [[ "$RUN_SNAPSHOT" == "1" ]]; then
  echo
  echo "=== Snapshot + LLM + execution plan ==="
  conda run -n "$CONDA_ENV" python -m robot_scene_pipeline.snapshot_pipeline \
    --use-tf \
    --tf-json "$TF_JSON" \
    --tf-point-mode optical-to-camera-link \
    --model "$MODEL" \
    --instruction "$INSTRUCTION" \
    --output-dir "$OUTPUT_DIR" \
    --compile-execution-plan \
    --place-offset-m "$PLACE_OFFSET_M" \
    --approach-height-m "$APPROACH_HEIGHT_M" \
    --pick-target-lift-m "$PICK_TARGET_LIFT_M" \
    --place-target-lift-m "$PLACE_TARGET_LIFT_M" \
    --left-right-axis "$LEFT_RIGHT_AXIS" \
    --left-direction-sign "$LEFT_DIRECTION_SIGN" \
    --front-back-axis "$FRONT_BACK_AXIS" \
    --front-direction-sign "$FRONT_DIRECTION_SIGN"
else
  echo
  echo "=== Reusing existing execution plan ==="
fi

if [[ ! -f "$PLAN_JSON" ]]; then
  echo "ERROR: execution plan not found: $PLAN_JSON" >&2
  exit 2
fi

echo
echo "=== Execution plan summary ==="
python3 - "$PLAN_JSON" "$PATH_MODE" <<'PY'
import json
import sys

path = sys.argv[1]
path_mode = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    plan = json.load(f)
print("plan:", path)
bad = []
for step in plan.get("steps", []):
    print(
        "step {step} {action} object={obj} ref={ref} rel={rel} status={status}".format(
            step=step.get("step"),
            action=step.get("action"),
            obj=step.get("object_label"),
            ref=step.get("reference_label"),
            rel=step.get("relative_position"),
            status=step.get("status"),
        )
    )
    print("  target  ", step.get("target_position_m"))
    print("  approach", step.get("approach_position_m"))
    action = step.get("action")
    if action in ("ask_user", "stop"):
        continue
    if step.get("status") != "planned":
        bad.append((step.get("step"), action, step.get("status"), "status is not planned"))
    elif not step.get("approach_position_m"):
        bad.append((step.get("step"), action, step.get("status"), "missing approach_position_m"))
    elif path_mode == "full" and not step.get("target_position_m"):
        bad.append((step.get("step"), action, step.get("status"), "missing target_position_m"))
if bad:
    print("\nERROR: execution plan is incomplete; refusing to start MoveIt/gripper.", file=sys.stderr)
    for item in bad:
        print("  step {} {} status={} {}".format(*item), file=sys.stderr)
    print("Regenerate the snapshot after making sure all referenced objects are visible and have valid depth.", file=sys.stderr)
    sys.exit(2)
PY

ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"

MOVEIT_CMD=(
  "$ROS_PYTHON" tools/moveit_plan_preview.py
  --plan-json "$PLAN_JSON"
  --all-approaches
  --path-mode "$PATH_MODE"
  --tool-offset-base "$TOOL_OFFSET_X" "$TOOL_OFFSET_Y" "$TOOL_OFFSET_Z"
  --velocity "$VELOCITY"
  --acceleration "$ACCELERATION"
)

if [[ "$ENABLE_GRIPPER" == "1" ]]; then
  MOVEIT_CMD+=(
    --enable-gripper
    --gripper-port "$GRIPPER_PORT"
    --gripper-force "$GRIPPER_FORCE"
    --gripper-speed "$GRIPPER_SPEED"
    --gripper-open-position "$GRIPPER_OPEN_POSITION"
    --gripper-close-position "$GRIPPER_CLOSE_POSITION"
  )
fi

if [[ "$EXECUTE" == "1" ]]; then
  MOVEIT_CMD+=(--execute)
fi

if [[ "$AUTO_YES" == "1" ]]; then
  MOVEIT_CMD+=(--yes)
fi

echo
echo "=== MoveIt execution ==="
echo "Command: ${MOVEIT_CMD[*]}"
echo "Do not run dh_demo.py at the same time; this script owns $GRIPPER_PORT when gripper is enabled."
if [[ "$AUTO_YES" == "1" ]]; then
  echo "AUTO_YES=1: motion and gripper confirmations are skipped."
else
  echo "AUTO_YES=0: each motion/gripper action asks for 'yes'."
fi
exec "${MOVEIT_CMD[@]}"
