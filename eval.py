import argparse
import csv
import multiprocessing
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from skimage import io

from handlers.transformations.rescale_for_eval import rescale_for_eval, to_tensor
from network.coordinate_network import CoordinatesRegressionNetwork


def evaluate(network):
    with open(os.path.join(os.getcwd(), "datasets", "MPII", "annotations", "test.csv")) as file:
        annotations = list(csv.reader(file, delimiter='\n'))

    individual_losses = [np.array([]) for i in range(16)]

    for i in range(len(annotations)):
        line = annotations[i][0].split(",")
        image = io.imread(os.path.join(os.getcwd(), "datasets", "MPII", "images", line[0]))

        joints = np.array([float(joint) for joint in line[1:]]).reshape([-1, 2])
        predicted_joints = get_joints(image, network)

        for j in range(16):
            euclidian_distance = np.linalg.norm(np.array(joints[j]) - predicted_joints[j])
            individual_losses[j] = np.append(individual_losses[j], euclidian_distance)

    individual_losses = [np.mean(individual_losses[i]) for i in range(len(individual_losses))]
    return individual_losses, np.mean(individual_losses)


def get_bounding_box_of_joints(joints):
    x_min = np.min(joints[:, 0])
    y_min = np.min(joints[:, 1])
    x_max = np.max(joints[:, 0])
    y_max = np.max(joints[:, 1])
    return x_min, y_min, x_max, y_max


def get_joints(image, network):
    height, width, _ = image.shape

    rescaled = rescale_for_eval(image, (224, 224))
    rescaled_image = rescaled['image']
    joints_fun = rescaled['joints_fun']
    image = to_tensor(image)
    image = image.unsqueeze(0)

    normalized_joints, _ = network(image)
    normalized_joints = normalized_joints[0].cpu().detach().numpy()
    joints = joints_fun(normalized_joints)
    return joints


def create_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-model', type=str)
    argument_parser.add_argument('-t7', type=str)
    argument_parser.add_argument('--input_size', type=int, default=224)
    argument_parser.add_argument('--batch_size', type=int, default=32)
    return argument_parser


def create_pytorch_device():
    device = torch.device("cuda:0")
    num_threads = (multiprocessing.cpu_count() // 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    return device, num_threads


def create_network(device, model, t7):
    net = CoordinatesRegressionNetwork(16, model).to(device)
    net = torch.nn.DataParallel(net).to(device)
    net.module.load_state_dict(torch.load(t7))
    net.eval()
    return net


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    device, num_threads = create_pytorch_device()
    net = create_network(device, args.model, args.t7)

    individual_losses, mean_loss = evaluate(net)
    print(individual_losses)
    print(mean_loss)
