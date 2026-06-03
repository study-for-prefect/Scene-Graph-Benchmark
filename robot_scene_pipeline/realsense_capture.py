import time

import numpy as np


def add_realsense_args(parser):
    parser.add_argument("--image-in", default="", help="Use an existing color image instead of capturing from RealSense.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--auto-profile", action="store_true", default=True)
    parser.add_argument("--no-auto-profile", action="store_false", dest="auto_profile")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--rs-reset", action="store_true")
    parser.add_argument("--rs-exposure", type=float, default=-1.0)
    parser.add_argument("--rs-gain", type=float, default=-1.0)


def configure_color_sensor(rs, profile, exposure, gain):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        if "rgb" not in name.lower():
            continue
        if exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        elif sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        if exposure >= 0 and sensor.supports(rs.option.exposure):
            sensor.set_option(rs.option.exposure, float(exposure))
        if gain >= 0 and sensor.supports(rs.option.gain):
            sensor.set_option(rs.option.gain, float(gain))
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 1)


def hardware_reset(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found.")
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print("Hardware-resetting RealSense serial={}...".format(serial), flush=True)
    devices[0].hardware_reset()
    time.sleep(8.0)


def list_realsense_devices(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    print("RealSense devices: {}".format(len(devices)), flush=True)
    for idx, dev in enumerate(devices):
        fields = {}
        for info in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.usb_type_descriptor,
            rs.camera_info.physical_port,
        ]:
            if dev.supports(info):
                fields[str(info)] = dev.get_info(info)
        print("device {}: {}".format(idx, fields), flush=True)


def stream_profiles(args):
    requested = (args.width, args.height, args.depth_width, args.depth_height, args.fps)
    candidates = [
        requested,
        (640, 480, 640, 360, 15),
        (640, 480, 480, 270, 15),
        (424, 240, 480, 270, 15),
        (640, 480, 640, 480, 30),
        (640, 480, 640, 360, 30),
        (640, 480, 480, 270, 30),
    ]
    output = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if not args.auto_profile:
            break
    return output


def read_aligned_rgbd(rs, pipeline, align, timeout_ms, warmup_frames):
    frame_bgr = None
    depth_frame = None
    for idx in range(max(0, warmup_frames) + 1):
        frames = pipeline.wait_for_frames(timeout_ms)
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue
        frame_bgr = np.asanyarray(color.get_data())
        depth_frame = depth
        if idx < warmup_frames:
            continue
        break
    if frame_bgr is None or depth_frame is None:
        raise RuntimeError("No aligned RGB-D frame received.")
    return frame_bgr, depth_frame


def capture_color_only(rs, args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    print(
        "Starting RealSense color-only fallback color={}x{}@{}".format(
            args.width, args.height, args.fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    try:
        frame_bgr = None
        for idx in range(max(0, args.warmup_frames) + 1):
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            color = frames.get_color_frame()
            if not color:
                continue
            frame_bgr = np.asanyarray(color.get_data())
            if idx < args.warmup_frames:
                continue
            break
        if frame_bgr is None:
            raise RuntimeError("No color frame received.")
        used_profile = {
            "color_width": args.width,
            "color_height": args.height,
            "depth_width": None,
            "depth_height": None,
            "fps": args.fps,
            "depth_available": False,
        }
        return frame_bgr, None, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)

    print(
        "Starting RealSense color={}x{}@{} depth={}x{}@{}".format(
            width, height, fps, depth_width, depth_height, fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    try:
        frame_bgr, depth_frame = read_aligned_rgbd(
            rs,
            pipeline,
            align,
            args.frame_timeout_ms,
            args.warmup_frames,
        )
        used_profile = {
            "color_width": width,
            "color_height": height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "fps": fps,
            "depth_available": True,
        }
        return frame_bgr, depth_frame, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd(args):
    import pyrealsense2 as rs

    list_realsense_devices(rs)
    if args.rs_reset:
        hardware_reset(rs)

    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        if width < args.width or height < args.height:
            print(
                "Skipping lower color profile {}x{} to preserve detector resolution.".format(
                    width, height
                ),
                flush=True,
            )
            continue
        try:
            return capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps)
        except RuntimeError as exc:
            error = "profile color={}x{}@{} depth={}x{}@{} failed: {}".format(
                width, height, fps, depth_width, depth_height, fps, exc
            )
            print(error, flush=True)
            errors.append(error)
            time.sleep(1.0)

    try:
        return capture_color_only(rs, args)
    except RuntimeError as exc:
        errors.append("color-only fallback failed: {}".format(exc))
    raise RuntimeError("All RealSense profiles failed:\n{}".format("\n".join(errors)))
