# TCP、桌面平面与物体 Yaw 标定流程

## 1. 必须区分的坐标变换

- `base_link <- camera_link`：手眼标定。决定相机点云能否正确进入机器人基坐标系。
- `tool0 -> TCP`：工具中心点标定。决定目标 TCP 位置如何换算成 MoveIt 的 `tool0` 目标。
- `table plane in base_link`：桌面平面估计。决定物体高度和桌面内的二维坐标基。
- `object yaw in table frame`：物体长轴相对桌面 x 轴的旋转角。

TCP 标定不会修正错误的手眼标定；桌面平面和 yaw 要进入 `base_link`，必须先保证手眼 TF 正确。

## 2. 真实 TCP 枢轴标定

在夹爪上安装刚性标定尖端。尖端必须代表实际希望控制的 TCP，例如两指闭合中心线上的抓取中心。让尖端始终接触同一个固定凹点，改变至少 8 到 12 个明显不同的工具姿态。

启动 UR 驱动和 `robot_state_publisher` 后运行：

```bash
/usr/bin/python3 tools/tcp_pivot_calibration.py \
  --base-frame base_link \
  --tool-frame tool0 \
  --sample-count 10 \
  --output /tmp/tcp_pivot_calibration.json
```

验收条件：

- `quality_pass=true`
- `rms_error_m <= 0.002`
- `condition_number <= 100`
- 最大误差尽量小于 `0.004 m`
- 各姿态必须有足够大的 roll/pitch/yaw 变化，不能只绕单一轴小幅转动

枢轴法只标定 TCP 平移，不标定 TCP 朝向。夹爪 TCP 朝向需要按机械结构定义，并通过实际夹持方向验证。

MoveIt 规划时使用真实 TCP：

```bash
/usr/bin/python3 tools/moveit_plan_preview.py \
  --plan-json /tmp/robot_scene_pipeline/robot_execution_plan.json \
  --tcp-calibration-json /tmp/tcp_pivot_calibration.json \
  --all-approaches
```

提供 `--tcp-calibration-json` 后，会使用：

```text
p_base_tool0 = p_base_tcp - R_base_tool0 * p_tool0_tcp
```

此时不再使用旧的固定基坐标补偿 `--tool-z-offset/--tool-offset-base`。

## 3. 验证手眼 TF

当前点云来自 RealSense optical frame，轴方向为 x 向右、y 向下、z 向前；当前管线使用：

```text
optical xyz -> camera_link xyz = [z, -x, -y]
```

启动动态 TF JSON 桥：

```bash
/usr/bin/python3 tools/tf_lookup_json.py \
  --base-frame base_link \
  --camera-frame camera_link \
  --output /tmp/scene_tf_base_camera.json
```

把一个固定标志点放在桌面上，从不同机器人观察姿态测量该点。转换后的 `base_link` 坐标应基本不变。若变化超过约 5 mm 到 10 mm，先重新做手眼标定，不要继续桌面/yaw 标定。

## 4. 桌面平面与尺寸/Yaw 估计

运行单帧点云估计：

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --tf-json /tmp/scene_tf_base_camera.json \
  --tf-point-mode optical-to-camera-link \
  --estimate-tabletop \
  --skip-llm \
  --output-dir /tmp/robot_scene_geometry
```

输出：

- `/tmp/robot_scene_geometry/tabletop_geometry.json`
- `/tmp/robot_scene_geometry/detector_objects_3d.json`
- `/tmp/robot_scene_geometry/private_scene_state.json`

桌面平面验收条件：

- `table_plane.frame == "base_link"`
- 法向量应接近 base 的 `+Z`，即 `normal[2]` 接近 `1`
- `inlier_ratio` 建议大于 `0.4`
- `rms_error_m` 建议小于 `0.005`
- 桌面高度 `origin_m[2]` 应与实测一致

物体输出字段：

- `dimensions_m = [length, width, height]`
- `table_yaw_deg`：物体长轴相对桌面 x 轴的角度，范围为 `[-90, 90)`
- `table_yaw_valid`：长宽比不足时为 false
- `footprint_aspect_ratio`：用于判断 yaw 稳定性
- `geometry_center_m`：点云物体中心
- `center_on_table_m`：物体中心投影到桌面的位置

对于圆柱、正方形或被严重遮挡的物体，yaw 本身不可观测或不稳定。默认长宽比小于 `1.20` 时，`table_yaw_valid=false`，不得把该 yaw 直接发送给机械臂。

## 5. 将物体 Yaw 接入 MoveIt

从最新点云场景中为长方形生成确定性的单物体抓取计划，不经过 LLM：

```bash
python3 tools/build_geometry_pick_plan.py \
  --private-state-json /tmp/robot_scene_geometry/private_scene_state.json \
  --object-label rectangle \
  --approach-height-m 0.12 \
  --output /tmp/robot_scene_geometry/rectangle_pick_plan.json
```

第一次只规划到物体上方，不执行、不下降、不闭合夹爪：

```bash
/usr/bin/python3 tools/moveit_plan_preview.py \
  --plan-json /tmp/robot_scene_geometry/rectangle_pick_plan.json \
  --orientation-mode object-yaw \
  --grasp-axis long \
  --yaw-sign positive \
  --yaw-offset-deg 0
```

在 RViz 中检查末端是否保持向下，并且绕 `base_link Z` 转到了物体方向。参数含义：

- `--grasp-axis long`：工具 yaw 参考轴对齐物体长轴。
- `--grasp-axis short`：在长轴 yaw 上增加 `90°`。
- `--yaw-sign negative`：实际旋转方向与识别 yaw 相反时使用。
- `--yaw-offset-deg`：补偿夹爪闭合方向与工具 yaw 参考轴之间的固定角度。
- `--invalid-yaw-fallback fixed`：正方形或圆形使用固定向下姿态，不使用不稳定 yaw。

先用两个明显不同角度的长方形确认 `--yaw-sign`。然后调整 `--grasp-axis` 和
`--yaw-offset-deg`，直到夹爪在物体上方的方向正确。

只执行到高处接近点进行观察时，不要传 `--path-mode full` 或
`--enable-gripper`：

```bash
/usr/bin/python3 tools/moveit_plan_preview.py \
  --plan-json /tmp/robot_scene_geometry/rectangle_pick_plan.json \
  --orientation-mode object-yaw \
  --grasp-axis long \
  --yaw-sign positive \
  --yaw-offset-deg 0 \
  --velocity 0.03 \
  --acceleration 0.03 \
  --execute
```

该命令仍会要求人工输入 `yes`。确认上方位置、旋转方向和高度都正确后，才能测试下降和夹取。

## 6. 诊断抓取中心偏移

保持同一个物体和同一个高处接近点，依次测试多个工具 yaw：

```bash
/usr/bin/python3 tools/moveit_plan_preview.py \
  --plan-json /tmp/robot_scene_geometry/rectangle_pick_plan.json \
  --orientation-mode object-yaw \
  --diagnostic-yaw-deg 0 45 90 \
  --velocity 0.02 \
  --acceleration 0.02 \
  --execute
```

该模式只支持高处接近点，禁止 `--yes`。每到一个角度，记录：

```text
误差 = 夹爪实际中心 - 物体目标中心
```

可以直接记录 `base_link X/Y` 毫米误差，也可以面对相机画面记录：

- `camera X`：画面向右为正。
- `camera Y`：画面向下为正。

相机画面测量示例：

```bash
python3 tools/analyze_grasp_offset.py \
  --camera-sample 0  12  -3 \
  --camera-sample 45  8   7 \
  --camera-sample 90 -2  12
```

基坐标测量示例：

```bash
python3 tools/analyze_grasp_offset.py \
  --sample 0  5 -10 \
  --sample 45 5 -10 \
  --sample 90 5 -10
```

输出分类：

- `base_fixed`：偏移不随夹爪旋转，使用基坐标补偿，并检查手眼平移。
- `tool_rotating`：偏移随夹爪旋转，属于 TCP 偏移，应测量或标定 `tool0 -> TCP`。
- `mixed`：两类偏移同时存在。

若改变机器人观察姿态后，同一物体的 base 坐标明显变化，则属于手眼标定问题，不能只用固定补偿解决。

## 7. 推荐验收顺序

1. 验证 UR 本体标定和 `base_link -> tool0` TF。
2. 验证或重做手眼标定，确认固定点的 base 坐标在不同观察姿态下保持稳定。
3. 完成 TCP 枢轴标定，用标定文件规划但不执行，确认换算后的 `tool0` 目标合理。
4. 空桌采集桌面平面，检查法向、桌面高度和 RMS。
5. 放置已知尺寸的长方体，比较点云尺寸与卡尺实测值。
6. 分别放置在 0、30、60、90 度，检查 `table_yaw_deg`；注意长轴 yaw 存在 180 度对称。
7. 最后再标定“物体长轴 yaw”到“夹爪闭合方向 yaw”的固定符号和角度偏置。

在第 7 步完成前，只输出 yaw，不直接执行带旋转的抓取。
