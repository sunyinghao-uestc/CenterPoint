"""
BEV visualization of CenterPoint predictions.

Follows the visualize_sample pattern from nuscenes/eval/detection/render.py,
rendering predictions (already in LiDAR frame) overlaid on the LiDAR point cloud.

Usage:
    python tools/vis.py \
        --prediction work_dirs/20260603/prediction.pkl \
        --version v1.0-mini \
        --data-root data/nuScenes \
        --out-dir work_dirs/20260603/bev_vis \
        --score-threshold 0.1
"""

import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

NUSC_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Colors matching nuScenes official detection colors
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


def main():
    parser = argparse.ArgumentParser(description="BEV visualization of predictions")
    parser.add_argument("--prediction", required=True, help="Path to prediction.pkl")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--data-root", default="data/nuScenes", help="nuScenes data root")
    parser.add_argument("--out-dir", default="work_dirs/bev_vis", help="Output directory for images")
    parser.add_argument("--score-threshold", type=float, default=0.15, help="Score threshold for predictions")
    parser.add_argument("--eval-range", type=float, default=50, help="BEV range in meters")
    parser.add_argument("--nsweeps", type=int, default=1, help="Number of lidar sweeps")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to render")
    parser.add_argument("--show-gt", action="store_true", help="Overlay ground truth boxes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load predictions (dict: sample_token -> {box3d_lidar, scores, label_preds, ...})
    with open(args.prediction, "rb") as f:
        predictions = pickle.load(f)

    # Load nuScenes
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    sample_tokens = list(predictions.keys())
    if args.max_samples:
        sample_tokens = sample_tokens[:args.max_samples]

    print(f"Rendering {len(sample_tokens)} samples to {args.out_dir}")

    for idx, sample_token in enumerate(sample_tokens):
        pred = predictions[sample_token]

        # Get sensor & pose records (same as visualize_sample)
        sample_rec = nusc.get("sample", sample_token)
        sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])

        # Get point cloud in LiDAR frame
        pc, _ = LidarPointCloud.from_file_multisweep(
            nusc, sample_rec, "LIDAR_TOP", "LIDAR_TOP", nsweeps=args.nsweeps
        )

        # Init axes
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Show point cloud (in LiDAR frame, view=np.eye(4))
        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / args.eval_range)
        ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

        # Show ego vehicle
        ax.plot(0, 0, "x", color="black")

        # Show GT boxes first (draw under predictions)
        if args.show_gt:
            _, gt_boxes, _ = nusc.get_sample_data(sd_record["token"])
            for gt_box in gt_boxes:
                c = np.array(nusc.explorer.get_color(gt_box.name)) / 255.0
                gt_box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=2)

        # Build prediction boxes (drawn on top of GT)
        box3d = pred["box3d_lidar"]
        scores = pred["scores"]
        labels = pred["label_preds"]

        mask = scores > args.score_threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]

        for box_tensor, score, label in zip(box3d, scores, labels):
            # box3d_lidar layout: [x, y, z, w, l, h, vx, vy, yaw]
            x, y, z, w, l, h, vx, vy, yaw_pred = box_tensor.tolist()
            class_name = NUSC_CLASS_NAMES[int(label)]
            color = np.array(DETECTION_COLORS[class_name]) / 255.0

            # Convert model yaw to nuScenes yaw convention
            # See _second_det_to_nusc_box in nusc_common.py:164
            yaw = -yaw_pred - np.pi / 2

            box = Box(
                center=[x, y, z],
                size=[w, l, h],
                orientation=Quaternion(axis=[0, 0, 1], radians=yaw),
                name=class_name,
            )
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

        # Limit visible range
        axes_limit = args.eval_range + 3
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        ax.set_title(sample_token)
        out_path = os.path.join(args.out_dir, f"{sample_token}.png")
        plt.savefig(out_path)
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Rendered {idx + 1}/{len(sample_tokens)}")

    print(f"Done. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
