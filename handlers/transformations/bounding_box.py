import numpy as np


def create_joints_bounding_box(left, right, top, bottom, full_height, full_width, ratio=0.15):
    width = right - left
    height = bottom - top

    translated_left = int(np.clip(left - ratio * width, 0, full_width))
    translated_right = int(np.clip(right + ratio * width, 0, full_width))
    translated_top = int(np.clip(top - ratio * height, 0, full_height))
    translated_bottom = int(np.clip(bottom + ratio * height, 0, full_height))
    return [translated_left, translated_top, translated_right, translated_bottom]