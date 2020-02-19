import numpy as np
import torch


class ToTensor(object):

    def __call__(self, sample):
        image, joints = sample['image'], sample['joints']
        height, width, _ = image.shape

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Equalize image RGB values, leaves Alpha untouched
        image[:, :, :3] = (image[:, :, :3] - mean) / (std)

        # Transforms to formats supported by Tensor objects
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        joints = torch.from_numpy(joints).float()

        return {
            'image': image,
            'joints': joints
        }