import os
import torch
import torch.utils.data as data
import torchvision.transforms as transform
from PIL import Image
from config import Config as cfg
from utils import get_path


class Dataset(data.Dataset):
    def __init__(self, dataset_root=None, transform_A=None, transform_B=None, mode='train'):
        super(Dataset, self).__init__()
        self.dataset_root = dataset_root
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.dir_A = os.path.join(dataset_root, mode + 'A')
        self.dir_B = os.path.join(dataset_root, mode + 'B')

        self.paths_A = sorted(os.listdir(self.dir_A))
        self.paths_B = sorted(os.listdir(self.dir_B))

        self.paths_A = get_path(self.dir_A, self.paths_A)
        self.paths_B = get_path(self.dir_B, self.paths_B)

        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)

    def __getitem__(self, index):
        index_A = self.paths_A[index % self.size_A]
        index_B = self.paths_B[index % self.size_B]
        image_A = Image.open(index_A).convert('RGB')
        image_B = Image.open(index_B).convert('RGB')

        if self.transform_A:
            image_A = self.transform_A(image_A)
        if self.transform_B:
            image_B = self.transform_B(image_B)

        return image_A, image_B

    def __len__(self):
        return max(self.size_A, self.size_B)


class SingleDataset(data.Dataset):
    def __init__(self, dataset_root=None, transform=None):
        super(SingleDataset, self).__init__()
        self.dataset_root = dataset_root
        self.transform = transform

        self.paths = sorted(os.listdir(dataset_root))

        self.paths = get_path(dataset_root, self.paths)

        self.size = len(self.paths)

    def __getitem__(self, index):
        index = self.paths[index % self.size]
        image = Image.open(index).convert('RGB')

        image = self.transform(image)

        return image

    def __len__(self):
        return self.size
