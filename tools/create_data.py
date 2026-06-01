import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    # 存一些类别的GT，构成一个数据库。在后面训练时将这些对象"粘贴"到当前场景中，增加场景的复杂度和多样性，有助于提升稀有类别和困难场景的检测性能
    if version == 'v1.0-trainval':
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
            virtual=virtual
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    fire.Fire()
