import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from .util import read_lines


class DenseFogDataset(Dataset):
    def __init__(self, root_dir: str, split_file: str, lidar_type: str):
        assert lidar_type in ['last', 'strongest']

        self.root_dir = root_dir
        self.lidar_type = lidar_type
        self.samples = read_lines(split_file)
        self.height, self.width = 192, 640

        self.min_depth, self.max_depth = 0.1, 100.0

    @staticmethod
    def crop_image(image: np.ndarray, depth: np.ndarray):
        image = image[288: -224, 96: -96, :]  # (512, 1728)
        depth = depth[288: -224, 96: -96]

        return image, depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # read color image
        image = cv2.imread(
            os.path.join(self.root_dir, 'cam_stereo_left_lut', sample + '.png')
        )[:, :, ::-1]

        # read depth image
        depth = np.load(
            os.path.join(self.root_dir, f'lidar_hdl64_{self.lidar_type}_stereo_left', sample + '.npz')
        )['arr_0'].astype(np.float32)

        # crop
        image, depth = self.crop_image(image, depth)

        # resize
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # format
        image = image.astype(np.float32) / 255.
        depth = depth.astype(np.float32)

        image = np.moveaxis(image, -1, 0).copy()

        return image, depth
