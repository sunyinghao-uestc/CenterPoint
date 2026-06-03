"""
BEV and camera-view visualization of CenterPoint predictions.

Follows the visualize_sample pattern from nuscenes/eval/detection/render.py
for BEV, and the vis_boxes pattern from calc_wheel_yaw.py for camera views.

Usage:
    # BEV only
    python tools/vis.py \
        --prediction work_dirs/20260603/prediction.pkl \
        --version v1.0-mini \
        --data-root data/nuScenes \
        --out-dir work_dirs/20260603/vis

    # BEV + GT overlay
    python tools/vis.py \
        --prediction work_dirs/20260603/prediction.pkl \
        --version v1.0-mini \
        --data-root data/nuScenes \
        --out-dir work_dirs/20260603/vis \
        --show-gt

    # BEV + camera views + GT
    python tools/vis.py \
        --prediction work_dirs/20260603/prediction.pkl \
        --version v1.0-mini \
        --data-root data/nuScenes \
        --out-dir work_dirs/20260603/vis \
        --show-gt --show-cameras
"""

import argparse
import os
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, view_points

CAM_SENSORS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

NUSC_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# nuScenes official detection colors (RGB)
DETECTION_COLORS = {
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "construction_vehicle": (0, 0, 230),
    "bus": (0, 60, 100),
    "trailer": (0, 0, 110),
    "barrier": (0, 0, 0),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
    "pedestrian": (220, 20, 60),
    "traffic_cone": (255, 0, 0),
}

# nuScenes official per-class evaluation distance thresholds (meters)
# See detection_cvpr_2019.json
CLASS_RANGES = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30,
}


def build_pred_boxes_lidar(box3d, scores, labels, score_threshold):
    """
    Build nuScenes Box objects in LiDAR frame from raw model output.

    Applies yaw conversion to nuScenes convention
    (see _second_det_to_nusc_box in nusc_common.py:164).

    Returns list of (box, class_name) tuples.
    """
    mask = scores > score_threshold
    box3d = box3d[mask]
    scores = scores[mask]
    labels = labels[mask]

    results = []
    for box_tensor, score, label in zip(box3d, scores, labels):
        # box3d_lidar layout: [x, y, z, w, l, h, vx, vy, yaw]
        x, y, z, w, l, h, vx, vy, yaw_pred = box_tensor.tolist()
        class_name = NUSC_CLASS_NAMES[int(label)]

        # Distance filter (matches filter_eval_boxes in nuscenes eval pipeline)
        dist = np.sqrt(x ** 2 + y ** 2)
        if dist > CLASS_RANGES[class_name]:
            continue

        yaw = -yaw_pred - np.pi / 2

        box = Box(
            center=[x, y, z],
            size=[w, l, h],
            orientation=Quaternion(axis=[0, 0, 1], radians=yaw),
            name=class_name,
        )
        results.append((box, class_name))
    return results


def boxes_lidar_to_ego(boxes, cs_record):
    """Convert boxes from LiDAR frame to ego frame (in-place)."""
    for box in boxes:
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))


def boxes_global_to_ego(boxes, pose_record):
    """Convert boxes from global frame to ego frame (in-place)."""
    for box in boxes:
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)


def render_boxes_on_camera(img, boxes_ego, cs_record, color_rgb):
    """
    Render boxes (ego frame) onto a camera image.

    Follows the vis_boxes pattern: ego -> sensor, filter by visibility,
    render with box.render_cv2. Boxes are copied so originals are not mutated.
    """
    cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    imsize = (img.shape[1], img.shape[0])
    color_bgr = color_rgb[::-1]  # RGB -> BGR for cv2

    for box in boxes_ego:
        box_sensor = box.copy()
        box_sensor.translate(-np.array(cs_record["translation"]))
        box_sensor.rotate(Quaternion(cs_record["rotation"]).inverse)
        if not box_in_image(box_sensor, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
            continue
        box_sensor.render_cv2(
            img, view=cam_intrinsic, normalize=True,
            colors=(color_bgr, color_bgr, color_bgr),
        )
    return img


def main():
    parser = argparse.ArgumentParser(description="BEV and camera visualization of predictions")
    parser.add_argument("--prediction", required=True, help="Path to prediction.pkl")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--data-root", default="data/nuScenes", help="nuScenes data root")
    parser.add_argument("--out-dir", default="work_dirs/vis", help="Output directory")
    parser.add_argument("--score-threshold", type=float, default=0.15, help="Score threshold")
    parser.add_argument("--eval-range", type=float, default=50, help="BEV range in meters")
    parser.add_argument("--nsweeps", type=int, default=1, help="Number of lidar sweeps")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to render")
    parser.add_argument("--show-gt", action="store_true", help="Overlay ground truth boxes")
    parser.add_argument("--show-cameras", action="store_true", help="Render 6 camera surround views")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bev_out_dir = os.path.join(args.out_dir, "bev")
    os.makedirs(bev_out_dir, exist_ok=True)
    if args.show_cameras:
        cam_out_dirs = {}
        for cam_name in CAM_SENSORS:
            cam_out_dirs[cam_name] = os.path.join(args.out_dir, cam_name)
            os.makedirs(cam_out_dirs[cam_name], exist_ok=True)

    with open(args.prediction, "rb") as f:
        predictions = pickle.load(f)

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    sample_tokens = list(predictions.keys())
    if args.max_samples:
        sample_tokens = sample_tokens[:args.max_samples]

    print(f"Rendering {len(sample_tokens)} samples to {args.out_dir}")

    for idx, sample_token in enumerate(sample_tokens):
        pred = predictions[sample_token]

        # --- Common data ---
        sample_rec = nusc.get("sample", sample_token)
        sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        # Build prediction boxes in LiDAR frame
        pred_boxes_lidar = build_pred_boxes_lidar(
            pred["box3d_lidar"], pred["scores"], pred["label_preds"],
            args.score_threshold,
        )
        pred_boxes = [b for b, _ in pred_boxes_lidar]
        pred_classes = [c for _, c in pred_boxes_lidar]

        # ================================================================
        # BEV visualization (LiDAR frame)
        # ================================================================
        pc, _ = LidarPointCloud.from_file_multisweep(
            nusc, sample_rec, "LIDAR_TOP", "LIDAR_TOP", nsweeps=args.nsweeps
        )

        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / args.eval_range)
        ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
        ax.plot(0, 0, "x", color="black")

        # GT on BEV (LiDAR frame, drawn first)
        if args.show_gt:
            _, gt_boxes_lidar, _ = nusc.get_sample_data(sd_record["token"])
            for gt_box in gt_boxes_lidar:
                c = np.array(nusc.explorer.get_color(gt_box.name)) / 255.0
                gt_box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=2)

        # Predictions on BEV (LiDAR frame, drawn on top)
        for box, class_name in zip(pred_boxes, pred_classes):
            color = np.array(DETECTION_COLORS[class_name]) / 255.0
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

        axes_limit = args.eval_range + 3
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.set_title(sample_token)
        out_path = os.path.join(bev_out_dir, f"{sample_token}.png")
        plt.savefig(out_path)
        plt.close()

        # ================================================================
        # Camera surround views
        # ================================================================
        if args.show_cameras:
            # Convert pred boxes: LiDAR -> ego (shared across all cameras)
            pred_boxes_ego = [b.copy() for b in pred_boxes]
            boxes_lidar_to_ego(pred_boxes_ego, cs_record)

            for cam_name in CAM_SENSORS:
                cam_sd_token = sample_rec["data"][cam_name]
                cam_sd_record = nusc.get("sample_data", cam_sd_token)
                cam_cs_record = nusc.get(
                    "calibrated_sensor", cam_sd_record["calibrated_sensor_token"]
                )
                cam_pose_record = nusc.get(
                    "ego_pose", cam_sd_record["ego_pose_token"]
                )

                # Load image
                img_path = os.path.join(args.data_root, cam_sd_record["filename"])
                img = np.asarray(Image.open(img_path)).copy()

                # GT on camera (drawn first)
                if args.show_gt:
                    gt_boxes_global = nusc.get_boxes(cam_sd_token)
                    gt_boxes_ego = [b.copy() for b in gt_boxes_global]
                    boxes_global_to_ego(gt_boxes_ego, cam_pose_record)
                    for gt_box in gt_boxes_ego:
                        c_rgb = tuple(nusc.explorer.get_color(gt_box.name))
                        render_boxes_on_camera(img, [gt_box], cam_cs_record, c_rgb)

                # Predictions on camera (drawn on top)
                for box_ego, class_name in zip(pred_boxes_ego, pred_classes):
                    c_rgb = DETECTION_COLORS[class_name]
                    render_boxes_on_camera(img, [box_ego], cam_cs_record, c_rgb)

                cam_out_path = os.path.join(
                    cam_out_dirs[cam_name], f"{sample_token}.jpg"
                )
                Image.fromarray(img).save(cam_out_path)

        if (idx + 1) % 10 == 0:
            print(f"  Rendered {idx + 1}/{len(sample_tokens)}")

    print(f"Done. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
