import csv
import os

import numpy as np
from skimage import io
from torch.utils.data import Dataset

from handlers.transformations.bounding_box import create_joints_bounding_box


class PoseDataset(Dataset):

    def __init__(self, annotated_file, transformations):
        with open(annotated_file) as file:
            self.annotations = list(csv.reader(file, delimiter='\n'))
        self.transformations = transformations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        # Loads single annotated item and corresponding image
        line = self.annotations[item][0].split(",")
        image = io.imread(os.path.join(os.getcwd(), "datasets", "MPII", "images", line[0]))
        height, width, _ = image.shape

        # Forms bounding box from annotated joints
        joints = np.array([float(joint) for joint in line[1:]]).reshape([-1, 2])
        x_min = np.min(joints[:, 0])
        y_min = np.min(joints[:, 1])
        x_max = np.max(joints[:, 0])
        y_max = np.max(joints[:, 1])
        bounds_left, bounds_top, bounds_right, bounds_bottom = create_joints_bounding_box(
            x_min, x_max, y_min, y_max, height, width)

        # Transforms image and joints to match new size
        image = image[bounds_top:bounds_bottom, bounds_left:bounds_right, :]
        joints = (joints - np.array([bounds_left, bounds_top])).flatten()

        # Transforms dataset item before returning it
        sample = {
            'image': image,
            'joints': joints
        }
        if self.transformations:
            sample = self.transformations(sample)
        return sample
