import json
import os
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


def find_class_to_id_map(data):
    class_map = {}
    for d in data:
        class_map[d[0]] = d[1]
    class_map = dict(sorted(class_map.items()))
    return class_map


ID_TO_CLASS = {
    0: "Airplane",
    1: "Bag",
    2: "Cap",
    3: "Car",
    4: "Chair",
    5: "Earphone",
    6: "Guitar",
    7: "Knife",
    8: "Lamp",
    9: "Laptop",
    10: "Motorbike",
    11: "Mug",
    12: "Pistol",
    13: "Rocket",
    14: "Skateboard",
    15: "Table",
}


class ShapeNetDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.files = self.load_splits()

    def load_splits(self):
        if self.mode == "train":
            splits_path = self.root_dir / "train_split.json"
        elif self.mode == "val":
            splits_path = self.root_dir / "val_split.json"
        else:
            raise KeyError
        assert os.path.isfile(splits_path)
        with open(splits_path, "r") as f:
            splits = json.load(f)
        return splits

    def __len__(self):
        return len(self.files)

    def load_pcl(self, path):
        pcl_path = self.root_dir / Path(path)
        assert os.path.isfile(pcl_path)
        pcl = np.load(pcl_path)
        return pcl

    def load_segm(self, path):
        segm_path = self.root_dir / Path(path)
        assert os.path.isfile(segm_path)
        segm = np.loadtxt(segm_path, dtype=int)
        segm = segm - 1  # start from 0
        return segm

    def sample_points(self, pcl, segm):
        assert pcl.shape[0] == segm.shape[0]
        n_points = self.cfg["DATA"]["n_points"]
        samples = np.random.choice(pcl.shape[0], size=n_points, replace=True)
        return pcl[samples], segm[samples]

    def center_and_normalize(self, points):
        mean = points.mean(axis=0)
        centered = points - mean

        max_dist = np.linalg.norm(centered, axis=1).max()
        normalized = centered / max_dist
        return normalized

    def random_rotation(self, points):
        rot = R.random()
        R_matrix = rot.as_matrix()
        points_rot = points @ R_matrix.T
        return points_rot

    def apply_augmentations(self, points):
        points = self.random_rotation(points)
        return points

    def __getitem__(self, idx):
        class_id, class_name, pcl_path, segm_path = self.files[idx]
        pcl_raw = self.load_pcl(pcl_path)
        segm_raw = self.load_segm(segm_path)
        pcl, segm = self.sample_points(pcl_raw, segm_raw)
        pcl = self.center_and_normalize(pcl)
        if self.mode == "train":
            pcl = self.apply_augmentations(pcl)
        labels = class_id if self.cfg["MODEL"]["type"] == "classification" else segm
        return pcl.astype(np.float32), labels
