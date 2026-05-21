# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CenterPoint is a 3D object detection and tracking framework using center points in bird-eye view (BEV). Published at CVPR 2021. Supports nuScenes and Waymo Open Dataset. Built on PyTorch with sparse convolution (spconv).

## Commands

### Build CUDA extensions (required before first use)

```bash
cd det3d/ops/iou3d_nms && python setup.py build_ext --inplace
cd det3d/ops/dcn && python setup.py build_ext --inplace  # optional, needs older PyTorch
```

### Data preparation

```bash
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10

# Waymo (convert tfrecord first, then create infos)
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1
```

### Training (distributed, 4 GPUs)

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH
```

### Testing (distributed, 4 GPUs)

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth
```

### Single-GPU speed test

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test
```

### Tracking

```bash
# nuScenes
python tools/nusc_tracking/pub_test.py --work_dir WORK_DIR_PATH --checkpoint DETECTION_PATH

# Waymo
python tools/waymo_tracking/test.py ...
```

## Architecture

### Registry system

All model components are registered via `det3d.utils.Registry` (inspired by MMCV). The registries are defined in `det3d/models/registry.py`: `READERS`, `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `DETECTORS`, `SECOND_STAGE`, `ROI_HEAD`. Components are decorated with `@REGISTRY.register_module` and built via `det3d/models/builder.py`.

### Data flow

```
Point Cloud
  → Reader (voxelization / pillar encoding)
  → Backbone (sparse 3D conv, e.g. SpMiddleResNetFHD)
  → Neck (2D BEV conv, e.g. RPN)
  → Bbox Head (CenterHead: heatmap + regression)
  → [optional] Second Stage (point-level feature refinement)
  → Detection output (3D boxes, classes, velocity)
```

### Key model components

- **Readers** (`det3d/models/readers/`): VoxelFeatureExtractorV3, PillarFeatureNet, DynamicVoxelEncoder — convert raw points to voxel/pillar features
- **Backbones** (`det3d/models/backbones/`): Sparse 3D conv backbones using spconv (SpMiddleResNetFHD)
- **Necks** (`det3d/models/necks/`): RPN — 2D convolutional neck that processes the BEV feature map after projecting sparse features to dense
- **Bbox Heads** (`det3d/models/bbox_heads/`): CenterHead — predicts class-specific heatmaps, 3D box attributes (size, orientation, offset, velocity)
- **Detectors** (`det3d/models/detectors/`): VoxelNet (single stage), PointPillars, TwoStageDetector
- **Second Stage** (`det3d/models/second_stage/`): Refines first-stage boxes using point features sampled from BEV feature map
- **ROI Heads** (`det3d/models/roi_heads/`): RoI-wise classification and box regression on second-stage features

### Config system

Configs are executable Python files (not YAML) at `configs/{nusc,waymo}/{voxelnet,pp}/`. A config defines:
- `model`: Dict with `type`, `reader`, `backbone`, `neck`, `bbox_head`
- `data`: Train/val/test dataset specs, pipeline stages, batch size
- `train_cfg`: Assigner settings (target_assigner, gaussian overlap, etc.)
- `test_cfg`: Post-processing (NMS, score threshold, pc_range)
- `optimizer` / `lr_config`: Adam with OneCycleLR
- Pipeline: `LoadPointCloudFromFile → LoadPointCloudAnnotations → Preprocess → Voxelization → AssignLabel → Reformat`

The `det3d/torchie/` module handles config parsing (`Config.fromfile()`), distributed training, checkpointing, and logging.

### Dataset pipeline

- `det3d/datasets/nuscenes/` — nuScenes dataset class, info creation, evaluation
- `det3d/datasets/waymo/` — Waymo dataset class, tfrecord converter, evaluation
- `det3d/datasets/pipelines/` — Data loading, preprocessing (augmentation), voxelization, and formatting

### Tracking

Simple greedy closest-point matching. Detection centers in the current frame are matched to transformed centers from the previous frame using Hungarian algorithm. Implemented in `tools/nusc_tracking/` and `tools/waymo_tracking/`.

### Custom ops

- `det3d/ops/iou3d_nms/` — Rotated 3D IoU and NMS (C++/CUDA extension)
- `det3d/ops/dcn/` — Deformable convolution (C++/CUDA extension, optional)
- `det3d/ops/point_cloud/` — BEV point cloud operations

## Python path

The repo root must be on `PYTHONPATH` so that `import det3d` works:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/CenterPoint"
```
