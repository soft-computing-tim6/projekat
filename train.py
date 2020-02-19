import argparse

import dsntnn
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

import multiprocessing
import numpy as np
import os

from tqdm import tqdm

from handlers.data_loaders.loader import get_test_loader, get_train_loader
from network.coordinate_network import CoordinatesRegressionNetwork
from test import test


def train(model, epochs, save_location, input_size, batch_size, learning_rate, t7):
    device, num_threads = create_pytorch_device()
    net = CoordinatesRegressionNetwork(16, model).to(device)
    net = torch.nn.DataParallel(net).to(device)

    min_loss = np.float("inf")
    if t7 != "":
        pre_trained = torch.load(t7)
        net.module.load_state_dict(pre_trained)

        for param in list(net.parameters()):
            param.requires_grad = True
        min_loss = np.mean(np.array(test(device, net, input_size, batch_size, num_threads)[0]))

    net = net.train()
    loader = get_train_loader(input_size, batch_size, num_threads)

    nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for epoch in range(epochs):
        training_average_loss = []
        training_coordinates_loss = []
        training_heatmaps_loss = []

        for _, sample in enumerate(tqdm(loader)):
            # Pass data to device
            images, joints = sample['image'].to(device), sample['joints'].to(device)

            coordinates, heatmaps = net(images)
            euclidian_loss, regularization_loss, average_loss = calculate_losses(coordinates, heatmaps, joints)

            # Clear memory
            del sample, images, joints, coordinates, heatmaps

            # Calculate gradients
            optimizer.zero_grad()
            average_loss.backward()
            optimizer.step()

            # Losses into
            training_average_loss.append(average_loss.item())
            training_coordinates_loss.append(torch.mean(euclidian_loss).item())
            training_heatmaps_loss.append(torch.mean(regularization_loss).item())

        if epoch % 2 == 0:
            test_average_loss, test_coordinates_loss, test_heatmaps_loss = \
                test(device, net, input_size, batch_size, num_threads)

            print("Epoch %s, validation average loss: %s" % (epoch, np.mean(np.array(test_average_loss))))
            print("Epoch %s, validation coordinates loss: %s" % (epoch, np.mean(np.array(test_coordinates_loss))))
            print("Epoch %s, validation heatmaps loss: %s" % (epoch, np.mean(np.array(test_heatmaps_loss))))

            if np.mean(np.array(test_average_loss)) < min_loss:
                min_loss = np.mean(np.array(test_average_loss))
                print("New lowest loss at: %s, saving model" % min_loss)
                torch.save(net.module.state_dict(), "%s/%s.t7" % (save_location, model))

            with open(os.path.join(save_location, "%s-log.txt" % model), 'a+') as log_file:
                log_file.writelines([
                    "Epoch %s \n" % epoch,
                    "\t- training average loss: %s\n" % np.mean(np.array(training_average_loss)),
                    "\t- training coordinates loss: %s\n" % np.mean(np.array(training_coordinates_loss)),
                    "\t- training heatmaps loss: %s\n" % np.mean(np.array(training_heatmaps_loss)),
                    "\t- validation average loss: %s\n" % np.mean(np.array(test_average_loss)),
                    "\t- validation coordinates loss: %s\n" % np.mean(np.array(test_coordinates_loss)),
                    "\t- validation heatmaps loss: %s\n" % np.mean(np.array(test_heatmaps_loss))
                ])
                log_file.flush()

        print("Epoch %s, training average loss: %s" % (epoch, np.mean(np.array(training_average_loss))))
        print("Epoch %s, training coordinates loss: %s" % (epoch, np.mean(np.array(training_coordinates_loss))))
        print("Epoch %s, training heatmaps loss: %s" % (epoch, np.mean(np.array(training_heatmaps_loss))))
        scheduler.step(1)


def create_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-model', type=str)
    argument_parser.add_argument('-epochs', type=int)
    argument_parser.add_argument('-save_location', type=str)
    argument_parser.add_argument('--input_size', type=int, default=224)
    argument_parser.add_argument('--batch_size', type=int, default=32)
    argument_parser.add_argument('--lr', type=float, default=1e-3)
    argument_parser.add_argument('--t7', type=str, default="")
    return argument_parser


def create_pytorch_device():
    # Enable CUDA necessary flags
    device = torch.device("cuda:0")
    num_threads = (multiprocessing.cpu_count() // 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    return device, num_threads


def calculate_losses(coordinates, heatmaps, joints):
    # Per-location euclidean losses
    euclidian_loss = dsntnn.euclidean_losses(coordinates, joints)

    # Per-location regularization losses
    regularization_loss = dsntnn.js_reg_losses(heatmaps, joints, 1.0)

    # Combined loss
    average_loss = dsntnn.average_loss(euclidian_loss + regularization_loss)
    return euclidian_loss, regularization_loss, average_loss


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    train(args.model, args.epochs, args.save_location, args.input_size, args.batch_size, args.lr, args.t7)
