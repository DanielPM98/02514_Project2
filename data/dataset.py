import torch
from torchvision import transforms
from torch.utils import data

from data.ph2 import Ph2
from data.drive import Drive

def get_dataloaders(root, name, resolution=256, batch_size=32, seed=42):
    IMG_SIZE = (resolution, resolution)
    generator = torch.Generator().manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST), # Resize all images to IMG_SIZExIMG_SIZEx3
        transforms.ToTensor(), # Transform to tensor
        # transforms.Normalize(mean=(0.5,), std=(0.5,)) # Normalize between [-1,1]
    ])
    if name == 'ph2':
        dataset  = Ph2(root, transform= transform)
        train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.64, 0.16, 0.2], generator)
    elif name == 'drive':
        dataset = Drive(root, train=True, transform=transform)
        train_dataset, val_dataset = data.random_split(dataset, [0.8, 0.2], generator)
        test_dataset = Drive(root, train=False, transform=transform)


    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_dataloader, val_dataloader, test_dataloader

