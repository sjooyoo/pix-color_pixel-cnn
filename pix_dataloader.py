#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.

    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """

    def __init__(self, root, transform=None):
        """Initializes image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('YCbCr')
        #grayscale_image = image.split()[0]
        #if self.transform is not None:
        #    image = self.transform(image)
        return image

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)


def data_loader(dataset, batch_size, num_workers=2):
    """Builds and returns Dataloader."""
    if dataset == 'imagenet':
        image_size = 64

        transform = transforms.Compose([
            # transforms.Scale(image_size),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

        traindir = './data/tiny-imagenet-200/train/'
        valdir = './data/tiny-imagenet-200/val/'

        train_dataset = dsets.ImageFolder(traindir, transform=transform)
        val_dataset = dsets.ImageFolder(valdir, transform=transform)
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)



# shuffle=False, True when training.
    '''elif dataset == 'cifar':
        image_size =32

        transform = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor()
        ])

        train_dataset = dsets.CIFAR10(root='./data/',
                                      train=True,
                                      transform=transform,
                                      download=True)
        val_dataset = dsets.CIFAR10(root='./data/',
                                    train=False,
                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers)'''

    return train_loader, val_loader, image_size
