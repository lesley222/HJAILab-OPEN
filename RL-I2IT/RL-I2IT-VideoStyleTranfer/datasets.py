import os
import torch
import torch.utils.data as data
import torchvision.transforms as transform
from PIL import Image


class PairDataset(data.Dataset):
    def __init__(self, dataset_root=None, transform_A=None, transform_B=None, mode='train'):
        super(PairDataset, self).__init__()
        self.dataset_root = dataset_root
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.dir_A = os.path.join(dataset_root, mode + 'A')
        self.dir_B = os.path.join(dataset_root, mode + 'B')

        self.paths_A = sorted(os.listdir(self.dir_A))
        self.paths_B = sorted(os.listdir(self.dir_B))

        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)

    def __getitem__(self, index):
        index_A = os.path.join(self.dir_A, self.paths_A[index % self.size_A])
        index_B = os.path.join(self.dir_B, self.paths_B[index % self.size_B])
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
    def __init__(self, dataset_folder_root=None, transform=None):
        super(SingleDataset, self).__init__()
        self.dataset_folder_root = dataset_folder_root
        self.transform = transform
        self.paths = self.get_folder_frames(self.dataset_folder_root)
        self.size = len(self.paths)-1

    def __getitem__(self, index):
        dir_name, video_name, img_name = self.paths[index]
        path = os.path.join(dir_name, img_name)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image, video_name

    def __len__(self):
        return self.size

    def get_folder_frames(self, folder):
        frame_names = []
        for video_name in sorted(os.listdir(folder)):
            dirs = os.listdir(os.path.join(folder, video_name))
            for fname in sorted(dirs):
                frame_names.append((os.path.join(folder, video_name), video_name, fname))
        return frame_names
