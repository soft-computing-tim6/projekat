import cv2
import numpy as np
import torch


def rescale_for_eval(image, output_size):
    image = image / 256.0
    height, width, _ = image.shape

    # Scale image by a shorter axis
    scale = min(float(output_size[0]) / float(height),
                float(output_size[1]) / float(width))
    scaled_height = int(height * scale)
    scaled_width = int(width * scale)
    image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    left_padding = (output_size[1] - scaled_width) // 2
    right_padding = (output_size[1] - scaled_width) - left_padding
    top_padding = (output_size[0] - scaled_height) // 2
    bottom_padding = (output_size[0] - scaled_height) - top_padding
    padding = ((top_padding, bottom_padding), (left_padding, right_padding))

    # Add padding to image
    mean = np.array([0.485, 0.456, 0.406])
    image = np.stack(
        [np.pad(image[:, :, color], padding, mode='constant', constant_values=mean[color]) for color in range(3)],
        axis=2)

    joints_fun = lambda x: ((((x.reshape([-1, 2]) + np.array([1.0, 1.0])) / 2.0 *
                np.array(output_size) - [left_padding, top_padding]) * 1.0 / np.array(
        [scaled_width, scaled_height]) * np.array([width, height])))

    return {
        'image': image,
        'joints_fun': joints_fun
    }


def to_tensor(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    return image
