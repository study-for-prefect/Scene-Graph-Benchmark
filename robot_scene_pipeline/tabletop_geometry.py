"""Estimate a tabletop plane and object footprint geometry from aligned depth."""

import math

import numpy as np


def add_tabletop_args(parser):
    parser.add_argument(
        "--estimate-tabletop",
        action="store_true",
        help="Estimate table plane plus per-object point-cloud dimensions and yaw.",
    )
    parser.add_argument("--plane-point-stride", type=int, default=4)
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.006)
    parser.add_argument("--plane-ransac-iterations", type=int, default=400)
    parser.add_argument("--plane-min-inliers", type=int, default=300)
    parser.add_argument("--plane-max-depth-m", type=float, default=2.0)
    parser.add_argument("--plane-min-up-alignment", type=float, default=0.70)
    parser.add_argument("--object-point-stride", type=int, default=2)
    parser.add_argument("--object-min-height-m", type=float, default=0.006)
    parser.add_argument("--object-max-height-m", type=float, default=0.50)
    parser.add_argument("--object-min-points", type=int, default=40)
    parser.add_argument("--yaw-min-aspect-ratio", type=float, default=1.20)


def deproject_depth_roi(depth_frame, intrinsics, roi=None, stride=1, max_depth_m=2.0):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    if roi is None:
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1
    else:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=float)

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    try:
        import pyrealsense2 as rs
    except ImportError:
        rs = None
    points = []
    for y in range(y1, y2 + 1, max(1, int(stride))):
        for x in range(x1, x2 + 1, max(1, int(stride))):
            depth = float(depth_frame.get_distance(x, y))
            if depth <= 0 or depth > float(max_depth_m):
                continue
            if rs is not None and hasattr(intrinsics, "model"):
                points.append(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth))
            else:
                points.append([(x - ppx) * depth / fx, (y - ppy) * depth / fy, depth])
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def transform_points(points, matrix=None, point_mode="optical-to-camera-link"):
    points = np.asarray(points, dtype=float)
    if not len(points):
        return points.reshape((-1, 3))
    if matrix is None:
        return points.copy()
    if point_mode == "optical-to-camera-link":
        points = np.column_stack([points[:, 2], -points[:, 0], -points[:, 1]])
    elif point_mode != "direct":
        raise ValueError("Unsupported point mode: {}".format(point_mode))
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    return homogeneous.dot(np.asarray(matrix, dtype=float).T)[:, :3]


def orient_normal(normal, orientation_hint):
    normal = np.asarray(normal, dtype=float)
    hint = np.asarray(orientation_hint, dtype=float) if orientation_hint is not None else None
    if hint is not None and np.dot(normal, hint) < 0:
        normal = -normal
    return normal


def fit_plane_ransac(
    points,
    distance_threshold_m=0.006,
    iterations=400,
    min_inliers=300,
    up_hint=None,
    min_up_alignment=0.0,
    orientation_hint=None,
    random_seed=7,
):
    points = np.asarray(points, dtype=float)
    if len(points) < max(3, int(min_inliers)):
        raise ValueError("Not enough points for table plane: {}.".format(len(points)))
    rng = np.random.RandomState(random_seed)
    best_mask = None
    best_count = 0
    best_rms = float("inf")
    up = None
    if up_hint is not None:
        up = np.asarray(up_hint, dtype=float)
        up /= np.linalg.norm(up)

    for _ in range(max(1, int(iterations))):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if up is not None and abs(float(np.dot(normal, up))) < float(min_up_alignment):
            continue
        d = -float(np.dot(normal, sample[0]))
        distances = np.abs(points.dot(normal) + d)
        mask = distances <= float(distance_threshold_m)
        count = int(np.count_nonzero(mask))
        if count < int(min_inliers):
            continue
        rms = float(np.sqrt(np.mean(distances[mask] ** 2)))
        if count > best_count or (count == best_count and rms < best_rms):
            best_mask = mask
            best_count = count
            best_rms = rms

    if best_mask is None:
        raise ValueError(
            "No table plane reached {} inliers. Check hand-eye TF, workspace ROI, and threshold.".format(min_inliers)
        )

    inlier_points = points[best_mask]
    origin = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - origin, full_matrices=False)
    normal = vh[-1]
    hint = up if up is not None else orientation_hint
    normal = orient_normal(normal, hint)
    d = -float(np.dot(normal, origin))
    distances = np.abs(points.dot(normal) + d)
    final_mask = distances <= float(distance_threshold_m)
    final_distances = distances[final_mask]
    return {
        "normal": normal,
        "d": d,
        "origin": origin,
        "inlier_mask": final_mask,
        "inlier_count": int(np.count_nonzero(final_mask)),
        "point_count": int(len(points)),
        "rms_error_m": float(np.sqrt(np.mean(final_distances ** 2))),
        "max_error_m": float(np.max(final_distances)),
    }


def make_plane_basis(normal, x_hint):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    candidates = [
        np.asarray(x_hint, dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    x_axis = None
    for candidate in candidates:
        projected = candidate - np.dot(candidate, normal) * normal
        norm = np.linalg.norm(projected)
        if norm > 1e-6:
            x_axis = projected / norm
            break
    if x_axis is None:
        raise ValueError("Cannot construct table plane basis.")
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis


def normalize_axis_yaw_rad(angle):
    angle = float(angle) % math.pi
    if angle >= math.pi / 2.0:
        angle -= math.pi
    return angle


def convex_hull_2d(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 1:
        return np.asarray(points, dtype=float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect_footprint(uv):
    hull = convex_hull_2d(uv)
    if len(hull) < 3:
        return None
    best = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        edge_norm = np.linalg.norm(edge)
        if edge_norm < 1e-9:
            continue
        u_axis = edge / edge_norm
        v_axis = np.array([-u_axis[1], u_axis[0]], dtype=float)
        u_values = uv.dot(u_axis)
        v_values = uv.dot(v_axis)
        u_extent = float(np.max(u_values) - np.min(u_values))
        v_extent = float(np.max(v_values) - np.min(v_values))
        area = u_extent * v_extent
        yaw_rad = math.atan2(u_axis[1], u_axis[0])
        length, width = u_extent, v_extent
        if width > length:
            length, width = width, length
            yaw_rad += math.pi / 2.0
        candidate = {
            "area": area,
            "length": length,
            "width": width,
            "yaw_rad": normalize_axis_yaw_rad(yaw_rad),
        }
        if best is None or candidate["area"] < best["area"]:
            best = candidate
    return best


def estimate_object_footprint(
    points,
    plane,
    min_height_m=0.006,
    max_height_m=0.50,
    min_points=40,
    yaw_min_aspect_ratio=1.20,
    use_min_area_rect=False,
):
    points = np.asarray(points, dtype=float)
    normal = plane["normal"]
    origin = plane["origin"]
    x_axis = plane["x_axis"]
    y_axis = plane["y_axis"]
    heights = (points - origin).dot(normal)
    mask = (heights >= float(min_height_m)) & (heights <= float(max_height_m))
    object_points = points[mask]
    object_heights = heights[mask]
    if len(object_points) < int(min_points):
        return None

    relative = object_points - origin
    uv = np.column_stack([relative.dot(x_axis), relative.dot(y_axis)])
    uv_center = np.median(uv, axis=0)
    centered = uv - uv_center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    long_axis_uv = eigenvectors[:, order[0]]
    if long_axis_uv[0] < 0:
        long_axis_uv = -long_axis_uv
    short_axis_uv = np.array([-long_axis_uv[1], long_axis_uv[0]], dtype=float)
    long_values = centered.dot(long_axis_uv)
    short_values = centered.dot(short_axis_uv)
    long_low, long_high = np.percentile(long_values, [2.0, 98.0])
    short_low, short_high = np.percentile(short_values, [2.0, 98.0])
    length = float(long_high - long_low)
    width = float(short_high - short_low)
    if width > length:
        length, width = width, length
        long_axis_uv = short_axis_uv
    aspect_ratio = length / max(width, 1e-9)
    yaw_rad = normalize_axis_yaw_rad(math.atan2(long_axis_uv[1], long_axis_uv[0]))
    yaw_source = "pca_long_axis"
    min_rect = min_area_rect_footprint(uv)
    if use_min_area_rect and min_rect is not None:
        length = min_rect["length"]
        width = min_rect["width"]
        aspect_ratio = length / max(width, 1e-9)
        yaw_rad = min_rect["yaw_rad"]
        yaw_source = "min_area_rect"
    height = float(np.percentile(object_heights, 98.0))
    center_plane = origin + uv_center[0] * x_axis + uv_center[1] * y_axis
    center_object = center_plane + float(np.median(object_heights)) * normal
    return {
        "point_count": int(len(object_points)),
        "dimensions_m": [length, width, height],
        "aspect_ratio": float(aspect_ratio),
        "yaw_rad": yaw_rad,
        "yaw_deg": math.degrees(yaw_rad),
        "yaw_valid": bool(aspect_ratio >= float(yaw_min_aspect_ratio)),
        "yaw_source": yaw_source,
        "center_on_plane_m": center_plane,
        "center_object_m": center_object,
        "height_percentiles_m": np.percentile(object_heights, [2.0, 50.0, 98.0]).tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }


def workspace_roi(detections):
    candidates = [
        det for det in detections if det.get("label") == "workspace" or det.get("is_workspace")
    ]
    if not candidates:
        return None
    workspace = max(candidates, key=lambda item: item.get("confidence", 0.0))
    return workspace.get("bbox")


def public_plane_payload(plane, frame):
    return {
        "frame": frame,
        "normal": plane["normal"].tolist(),
        "d": float(plane["d"]),
        "origin_m": plane["origin"].tolist(),
        "x_axis": plane["x_axis"].tolist(),
        "y_axis": plane["y_axis"].tolist(),
        "inlier_count": plane["inlier_count"],
        "point_count": plane["point_count"],
        "inlier_ratio": float(plane["inlier_count"]) / max(1, plane["point_count"]),
        "rms_error_m": plane["rms_error_m"],
        "max_error_m": plane["max_error_m"],
    }


def attach_tabletop_geometry(detections, depth_frame, intrinsics, args, transform_matrix=None):
    for det in detections:
        det["pointcloud_geometry_valid"] = False
        det["dimensions_m"] = None
        det["table_yaw_deg"] = None
        det["table_yaw_valid"] = False

    if depth_frame is None or intrinsics is None:
        raise ValueError("Tabletop estimation requires aligned depth and color intrinsics.")

    plane_points_optical = deproject_depth_roi(
        depth_frame,
        intrinsics,
        roi=workspace_roi(detections),
        stride=args.plane_point_stride,
        max_depth_m=args.plane_max_depth_m,
    )
    plane_points = transform_points(
        plane_points_optical,
        matrix=transform_matrix,
        point_mode=getattr(args, "tf_point_mode", "optical-to-camera-link"),
    )
    if not len(plane_points):
        raise ValueError("No valid depth points were available for table plane estimation.")
    in_base = transform_matrix is not None
    frame = args.base_frame if in_base else "camera_optical_frame"
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float) if in_base else None
    orientation_hint = up_hint if in_base else -np.mean(plane_points, axis=0)
    plane = fit_plane_ransac(
        plane_points,
        distance_threshold_m=args.plane_distance_threshold_m,
        iterations=args.plane_ransac_iterations,
        min_inliers=args.plane_min_inliers,
        up_hint=up_hint,
        min_up_alignment=args.plane_min_up_alignment if in_base else 0.0,
        orientation_hint=orientation_hint,
    )
    x_hint = np.array([1.0, 0.0, 0.0], dtype=float)
    plane["x_axis"], plane["y_axis"] = make_plane_basis(plane["normal"], x_hint)

    for det in detections:
        if det.get("label") == "workspace" or det.get("is_workspace"):
            continue
        object_points_optical = deproject_depth_roi(
            depth_frame,
            intrinsics,
            roi=det.get("bbox"),
            stride=args.object_point_stride,
            max_depth_m=args.plane_max_depth_m,
        )
        object_points = transform_points(
            object_points_optical,
            matrix=transform_matrix,
            point_mode=getattr(args, "tf_point_mode", "optical-to-camera-link"),
        )
        geometry = estimate_object_footprint(
            object_points,
            plane,
            min_height_m=args.object_min_height_m,
            max_height_m=args.object_max_height_m,
            min_points=args.object_min_points,
            yaw_min_aspect_ratio=args.yaw_min_aspect_ratio,
            use_min_area_rect=any(
                value in str(det.get("label") or "").lower()
                for value in ("square", "rectangle")
            ),
        )
        if geometry is None:
            continue
        det["pointcloud_geometry_valid"] = True
        det["pointcloud_point_count"] = geometry["point_count"]
        det["dimensions_m"] = geometry["dimensions_m"]
        det["footprint_aspect_ratio"] = geometry["aspect_ratio"]
        det["table_yaw_deg"] = geometry["yaw_deg"]
        det["table_yaw_rad"] = geometry["yaw_rad"]
        det["table_yaw_valid"] = geometry["yaw_valid"]
        det["table_yaw_source"] = geometry["yaw_source"]
        det["geometry_frame"] = frame
        det["geometry_center_m"] = geometry["center_object_m"].tolist()
        det["center_on_table_m"] = geometry["center_on_plane_m"].tolist()
        det["height_percentiles_m"] = geometry["height_percentiles_m"]

    return detections, public_plane_payload(plane, frame)
