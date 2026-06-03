"""Minimal RealSense probe without model loading.

Use this to separate camera/USB/SDK problems from detector problems.
"""

import argparse
import os
import time

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Probe RealSense color/depth frames.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--disable-depth", action="store_true")
    parser.add_argument("--align-depth", action="store_true", help="Align depth to color, matching the main pipeline.")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--out", default="/tmp/realsense_probe_color.jpg")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def list_devices(rs):
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


def configure_color_sensor(rs, profile):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        print("sensor: {}".format(name), flush=True)
        if "rgb" not in name.lower():
            continue
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("enabled RGB auto exposure", flush=True)
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 1)
            print("enabled RGB auto white balance", flush=True)


def save_frame(path, frame):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(path, frame)
    mean = np.mean(frame.reshape(-1, frame.shape[-1]), axis=0)
    print("saved {} size={} mean_bgr={}".format(path, frame.shape, [round(float(v), 2) for v in mean]), flush=True)


def main():
    args = parse_args()
    import pyrealsense2 as rs

    list_devices(rs)
    if args.reset:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device found for reset.")
        print("hardware reset...", flush=True)
        devices[0].hardware_reset()
        time.sleep(8.0)
        list_devices(rs)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.disable_depth:
        config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)

    print(
        "starting streams color={}x{}@{} depth={}".format(
            args.width,
            args.height,
            args.fps,
            "off" if args.disable_depth else "{}x{}@{}".format(args.depth_width, args.depth_height, args.fps),
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile)
    align = rs.align(rs.stream.color) if args.align_depth and not args.disable_depth else None

    try:
        if not args.no_window:
            cv2.namedWindow("RealSense Probe", cv2.WINDOW_NORMAL)
        got = 0
        for idx in range(args.frames):
            print("waiting frame {}/{}...".format(idx + 1, args.frames), flush=True)
            try:
                frames = pipeline.wait_for_frames(args.timeout_ms)
                if align is not None:
                    frames = align.process(frames)
            except RuntimeError as exc:
                print("timeout: {}".format(exc), flush=True)
                continue
            color = frames.get_color_frame()
            if not color:
                print("no color frame", flush=True)
                continue
            if not args.disable_depth:
                depth = frames.get_depth_frame()
                if not depth:
                    print("no depth frame", flush=True)
                    continue
                center_depth = float(depth.get_distance(args.width // 2, args.height // 2))
                print("center_depth_m={:.4f}".format(center_depth), flush=True)
            frame = np.asanyarray(color.get_data())
            got += 1
            save_frame(args.out, frame)
            if not args.no_window:
                cv2.imshow("RealSense Probe", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
        print("received color frames: {}".format(got), flush=True)
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
