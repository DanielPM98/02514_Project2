import torch
from PIL import Image
from torch.utils.data import Dataset

import os


class Drive(Dataset):
    def __init__(self, root, train, transform = None):
        self.transform = transform

        self.root = os.path.join(root, 'training' if train else 'test')

        self.image_paths = []
        self.label_paths = []
        for root_, _, files in os.walk(self.root):
            if root_.endswith('images'):
                [self.image_paths.append(os.path.join(root_, file_)) for file_ in files]
            if root_.endswith('mask'):
                [self.label_paths.append(os.path.join(root_, file_)) for file_ in files]

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform is not None:
            label = self.transform(label)
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample