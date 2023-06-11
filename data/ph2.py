import torch
from PIL import Image
from torch.utils.data import Dataset

import os

class Ph2(Dataset):
    def __init__(self, root, transform = None):
        self.transform = transform
        self.root = root

        self.image_paths = []
        self.label_paths = []

        for root_, _, files in os.walk(self.root):
            if root_.endswith('_Dermoscopic_Image'):
                self.image_paths.append(os.path.join(root_, files[0]))
            if root_.endswith('_lesion'):
                self.label_paths.append(os.path.join(root_, files[0]))

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
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
    
    def update_transform(self, new_transform):
        self.transform = new_transform
