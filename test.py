import argparse
import multiprocessing
import os

import dsntnn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from handlers.data_loaders.loader import get_test_loader
from network.coordinate_network import CoordinatesRegressionNetwork


def test(device, network, input_size, batch_size, num_threads):
    loader = get_test_loader(input_size, batch_size, num_threads)

    sample_average_loss = []
    sample_coordinates_loss = []
    sample_heatmaps_loss = []

    with torch.no_grad():
        for _, sample in enumerate(tqdm(loader)):
            images, joints = sample['image'].to(device), sample['joints'].to(device)
            coordinates, heatmaps = network(images)

            euclidian_loss, regularization_loss, average_loss = calculate_losses(coordinates, heatmaps, joints)
            del sample, images, joints, coordinates, heatmaps

            sample_average_loss.append(average_loss.item())
            sample_coordinates_loss.append(torch.mean(euclidian_loss).item())
            sample_heatmaps_loss.append(torch.mean(regularization_loss).item())

            return sample_average_loss, sample_coordinates_loss, sample_heatmaps_loss


def create_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-model', type=str)
    argument_parser.add_argument('--input_size', type=int, default=224)
    argument_parser.add_argument('--batch_size', type=int, default=32)
    argument_parser.add_argument('--t7', type=str, default="")
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

    if t7 != "":
        pre_trained = torch.load(t7)
        net.module.load_state_dict(pre_trained)

        for param in list(net.parameters()):
            param.requires_grad = True
    return net


def calculate_losses(coordinates, heatmaps, joints):
    euclidian_loss = dsntnn.euclidean_losses(coordinates, joints)
    regularization_loss = dsntnn.js_reg_losses(heatmaps, joints, 1.0)
    average_loss = dsntnn.average_loss(euclidian_loss + regularization_loss)
    return euclidian_loss, regularization_loss, average_loss


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    device, num_threads = create_pytorch_device()
    net = create_network(device, args.model, args.t7)

    sample_average_loss, sample_coordinates_loss, sample_heatmaps_loss = \
        test(device, net, args.input_size, args.batch_size, num_threads)
    print("Average loss: " + np.mean(np.array(sample_average_loss)))
    print("Coordinates loss: " + np.mean(np.array(sample_coordinates_loss)))
    print("Heatmaps loss:" + np.mean(np.array(sample_heatmaps_loss)))