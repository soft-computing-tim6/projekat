import cv2
import torch
import argparse

from handlers.visual.display_pose import display_pose
from handlers.transformations.rescale_for_eval import rescale_for_eval, to_tensor
from network.coordinate_network import CoordinatesRegressionNetwork


def create_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-model', type=str)
    argument_parser.add_argument('-t7',    type=str)
    return argument_parser


def get_joints(image, net):
    rescaled = rescale_for_eval(image, [224, 224])
    rescaled_image = rescaled['image']
    joints_fun = rescaled['joints_fun']
    image = to_tensor(rescaled_image)
    image = image.unsqueeze(0)

    normalized_joints = net(image)
    normalized_joints = normalized_joints[0].detach().numpy()
    joints = joints_fun(normalized_joints).astype(int)
    return joints


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    net = CoordinatesRegressionNetwork(16, args.model).to("cpu")
    net.load_state_dict(torch.load(args.t7, map_location=lambda storage, loc: storage))
    net.eval()
    
    cam = cv2.VideoCapture(0)
    _, image = cam.read()

    while True:
        _ , image = cam.read()
        joints = get_joints(image, net)
        image = display_pose(image, joints)
        cv2.imshow('Mom\'s camera', image)
        if cv2.waitKey(1) == 27: # ESC
            break
    cv2.destroyAllWindows()
