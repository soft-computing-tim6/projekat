import argparse

import cv2
import torch

from handlers.transformations.rescale_for_eval import rescale_for_eval, to_tensor
from handlers.visual.display_pose import display_pose
from network.coordinate_network import CoordinatesRegressionNetwork


def create_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-model', type=str)
    argument_parser.add_argument('-t7',    type=str)
    argument_parser.add_argument('-img',   type=str)
    return argument_parser


def get_joints(image, model, t7):
    rescaled = rescale_for_eval(image, [224, 224])
    rescaled_image = rescaled['image']
    joints_fun = rescaled['joints_fun']
    image = to_tensor(rescaled_image)
    image = image.unsqueeze(0)

    net = CoordinatesRegressionNetwork(16, model).to("cpu")
    net.load_state_dict(torch.load(t7, map_location=lambda storage, loc: storage))
    net.eval()

    normalized_joints = net(image)
    normalized_joints = normalized_joints[0].detach().numpy()
    joints = joints_fun(normalized_joints).astype(int)
    return joints


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    image = cv2.imread(args.img)
    joints = get_joints(image, args.model, args.t7)
    image = display_pose(image, joints)
    cv2.imwrite("out.png", image)




