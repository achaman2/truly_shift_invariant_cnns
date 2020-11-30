# %%
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

from torch.utils.data import Dataset
import random


import os

def set_random_seeds(seed):
    
    if seed is not None:
    
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)
        
    else:
        return

    
def get_dataloaders_circular(batch_size, dataset, dataset_path, train_split, base_center_crop, 
                   pin_memory, num_workers, worker_init_fn, data_loader_seed ):
    
    
    if dataset == 'cifar10':
        
        cifar_mean = np.array([0.49141738, 0.48219556, 0.44662726])
        cifar_std = np.array([0.24703224, 0.24348514, 0.26158786])
        
        if data_loader_seed is not None:
            set_random_seeds(data_loader_seed)
        
        
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                               transforms.CenterCrop(size = base_center_crop),
                                               transforms.ToTensor(),
                                               transforms.Normalize((cifar_mean), (cifar_std)),
                                              ])
        
        val_transform = transforms.Compose([transforms.CenterCrop(size = base_center_crop),
                                               transforms.ToTensor(),
                                               transforms.Normalize((cifar_mean), (cifar_std)),
                                              ])
                                               
        print('==> Preparing CIFAR-10 dataset..')
        train_set = torchvision.datasets.CIFAR10(root = dataset_path, train = True, 
                                                 download = True, transform = train_transform)
        
        validation_set = torchvision.datasets.CIFAR10(root = dataset_path, train = True, 
                                                 download = True, transform = val_transform)

        
#         now start train-val split
        split_len_train = int(train_split*len(train_set))
        split_len_val = len(train_set) - split_len_train
        
        indices = np.arange(len(train_set))
        
        if data_loader_seed is not None:
            set_random_seeds(data_loader_seed)
        
        np.random.shuffle(indices)
        
        train_indices, valid_indices = indices[0:split_len_train], indices[split_len_train : split_len_train + split_len_val]
        
        train_set = torch.utils.data.Subset(train_set, train_indices)
        validation_set = torch.utils.data.Subset(validation_set, valid_indices)
        test_set = torchvision.datasets.CIFAR10(root = dataset_path, train = False, 
                                                 download = True, transform = val_transform)
        
        data_loaders = {}
        data_loaders = {'train': torch.utils.data.DataLoader(
            train_set, batch_size = batch_size, shuffle = True, num_workers=num_workers,
               pin_memory=pin_memory, worker_init_fn = worker_init_fn),
                        
                        'val': torch.utils.data.DataLoader(
            validation_set, batch_size = batch_size, shuffle = False, num_workers=num_workers,
               pin_memory=pin_memory, worker_init_fn = worker_init_fn),
                        
                        'test': torch.utils.data.DataLoader(
            test_set, batch_size = batch_size, shuffle = False, num_workers=num_workers,
               pin_memory=pin_memory, worker_init_fn = worker_init_fn)}

        
    
        return data_loaders
    
    
    
    
    
