import os

from torch.utils.data import DataLoader
from torchvision import transforms

from handlers.datasets.pose_dataset import PoseDataset
from handlers.transformations.augmentation import Augmentation
from handlers.transformations.rescale import Rescale
from handlers.transformations.to_tensor import ToTensor


def get_test_loader(input_size, batch_size, number_of_threads):
    dataset = PoseDataset(
        os.path.join(os.getcwd(), 'datasets', 'MPII', 'annotations', 'test.csv'),
        transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor()
        ])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=number_of_threads)


def get_train_loader(input_size, batch_size, number_of_threads):
    dataset = PoseDataset(
        os.path.join(os.getcwd(), 'datasets', 'MPII', 'annotations', 'validation.csv'),
        transforms.Compose([
            Augmentation(),
            Rescale((input_size, input_size)),
            ToTensor()
        ])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_threads)

