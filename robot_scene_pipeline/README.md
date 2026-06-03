# Robot Scene Pipeline

This directory contains the split single-frame robot scene pipeline:

- `snapshot_pipeline.py`: main orchestration.
- `realsense_capture.py`: RealSense RGB-D snapshot capture.
- `detector_runtime.py`: custom detector runtime.
- `depth_geometry.py`: depth-to-3D and scene state helpers.
- `tf_transform.py`: TF lookup and camera-to-base point transform.
- `llm_scene_reasoner.py`: LLM prompt and Ollama call.

## Static TF

Publish the hand-eye calibration result in a terminal:

```bash
ros2 run tf2_ros static_transform_publisher \
  0.08599923 -0.03025062 0.00174066 \
  -0.706447 -0.01772374 -0.70735528 0.01634021 \
  wrist_3_link camera_link
```

Verify the TF chain after the robot driver or `robot_state_publisher` is running:

```bash
python -m robot_scene_pipeline.tf_transform \
  --base-frame base_link \
  --camera-frame camera_link
```

The `--camera-frame` must match the coordinate convention of the 3D points from
RealSense deprojection. If the calibration uses ROS optical-frame axes, publish
and use an optical frame such as `camera_color_optical_frame`.

## Offline Detector Test

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --image-in input_dir/1.jpg \
  --output-dir /tmp/robot_scene_pipeline_test \
  --device cpu \
  --skip-llm
```

## RealSense + LLM

Capture immediately:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Wait for manual confirmation before capture:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --capture-trigger enter \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Use TF to add base-frame coordinates:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --base-frame base_link \
  --camera-frame camera_link \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```
