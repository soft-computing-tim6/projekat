import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class Augmentation(object):

    def __call__(self, sample):
        image, joints = sample['image'], sample['joints'].reshape([-1, 2])

        sequential_transform = iaa.Sequential(
            [
                iaa.Sometimes(0.3, iaa.CropAndPad(
                    percent=(-0.3, 0.3),
                    pad_mode=["edge"],
                    keep_size=False
                )),
                iaa.Sometimes(0.3, iaa.Affine(
                    scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},
                    translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                    rotate=(-45, 45),
                    shear=(-5, 5),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                iaa.SomeOf((0, 3),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur(),
                                   iaa.MedianBlur(),
                                   iaa.MotionBlur(k=4, angle=[-30, 30])
                               ]),
                               iaa.OneOf([
                                   iaa.AdditiveGaussianNoise(
                                       scale=(0.0, 12.75),
                                       per_channel=0.5
                                   ),
                                   iaa.AdditivePoissonNoise(
                                       lam=(0.0, 7.0),
                                       per_channel=True
                                   )]
                               ),
                               iaa.OneOf([
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   iaa.Multiply((0.5, 1), per_channel=0.5),
                                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
                               ])
                           ],
                           random_order=True)
            ],
            random_order=True
        )

        sequential_deterministic = sequential_transform.to_deterministic()
        image_augment = sequential_deterministic.augment_images([image])[0]
        keypoints_augment = sequential_deterministic.augment_keypoints([joints_to_keypoints(image, joints)])[0]

        return {
            'image': image_augment,
            'joints': keypoints_to_joints(keypoints_augment)
        }


def joints_to_keypoints(image, joints):
    keypoints = []
    for row in range(int(joints.shape[0])):
        x = joints[row, 0]
        y = joints[row, 1]
        keypoints.append(ia.Keypoint(x, y))
    return ia.KeypointsOnImage(keypoints, image.shape)


def keypoints_to_joints(augmented_keypoints):
    joints_of_person = []
    for i, keypoint in enumerate(augmented_keypoints.keypoints):
        x, y = keypoint.x, keypoint.y
        joints_of_person.append(np.array(x).astype(np.float32))
        joints_of_person.append(np.array(y).astype(np.float32))
    return np.array(joints_of_person).reshape([-1, 2])