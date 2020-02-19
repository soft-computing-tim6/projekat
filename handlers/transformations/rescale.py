import cv2
import numpy as np


class Rescale(object):

    def __init__(self, output_size: (int, tuple)):
        self.output_size = output_size

    def __call__(self, sample):
        image, joints = sample['image'] / 256.0, sample['joints']
        height, width, _ = image.shape

        # Scale image by a shorter axis
        scale = min(float(self.output_size[0]) / float(height),
                    float(self.output_size[1]) / float(width))
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        left_padding = (self.output_size[1] - scaled_width) // 2
        right_padding = (self.output_size[1] - scaled_width) - left_padding
        top_padding = (self.output_size[0] - scaled_height) // 2
        bottom_padding = (self.output_size[0] - scaled_height) - top_padding
        padding = ((top_padding, bottom_padding), (left_padding, right_padding))

        # Add padding to image
        mean = np.array([0.485, 0.456, 0.406])
        image = np.stack(
            [np.pad(image[:, :, color], padding, mode='constant', constant_values=mean[color]) for color in range(3)],
            axis=2)

        # Transform annotations to fit new image size
        joints = (joints.reshape([-1, 2]) / np.array([width, height]) * np.array([scaled_width, scaled_height]))
        joints += [left_padding, top_padding]
        joints = (joints * 2 + 1) / self.output_size - 1

        return {
            'image': image,
            'joints': joints
        }