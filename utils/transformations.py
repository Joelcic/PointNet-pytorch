import numpy as np
import torch
from torchvision import transforms

def default_transforms(train=True):
    """
    Returns a composed set of transformations to be applied to point cloud data.
    If `train` is True, the returned transform includes data augmentation steps
    such as random rotation around the Z-axis and Gaussian noise addition.
    If False, only normalization is applied.
    :param train: (bool) Whether the transformations are for training (default is True).
    :return:
        transform: (torchvision.transforms.Compose) A composition of point cloud transforms.
    """
    if train:
        return transforms.Compose([
            Normalize(),
            RandomRotateZ(),
            GaussianNoice(),
        ])
    else:
        return transforms.Compose([
            Normalize(),
        ])

class Normalize(object):
    def __call__(self, pointcloud):
        pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        pointcloud /= np.max(np.linalg.norm(pointcloud, axis=1))
        return pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)

class RandomRotateZ(object):
    def __call__(self, pointcloud):
        theta = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return pointcloud @ rotation_matrix.T  # Matrix multiplication

class GaussianNoice(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        noise = np.clip(self.sigma * np.random.randn(*pointcloud.shape), -self.clip, self.clip)
        return pointcloud + noise

