import numpy as np
from os import path

from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset as TorchDataset
import torchvision
import torchvision.transforms as torchvision_transforms

dataset_dir = ''

class RandomAccessTorchDataset(TorchDataset):
    def __init__(self, dataset, part, center_crop, reshape=None):
        self.dataset = dataset
        self.part = part
        self.center_crop = center_crop
        self.reshape = reshape or (
            (None,) * self.dataset.get_num_elements_per_sample())

    def __len__(self):
        return self.dataset.get_len(self.part)

    def __getitem__(self, idx):
        arrays = self.dataset.get_sample(idx, part=self.part)
        mult_elem = isinstance(arrays, tuple)
        if not mult_elem:
            arrays = (arrays,)
        tensors = []
        for arr, s in zip(arrays, self.reshape):
            t = torch.from_numpy(np.asarray(arr))
            if s is not None:
                t = t.view(*s)
            
            t = torchvision_transforms.functional.center_crop(img = t, output_size = self.center_crop)



            
            tensors.append(t)
        return tuple(tensors) if mult_elem else tensors[0]


def get_dataloaders_ct(batch_size, num_workers, cache_dir, dataset_image_center_crop = 362, include_validation = True, IMPL = 'astra_cuda', **kwargs):
    
    if include_validation:
        parts = ['train', 'validation', 'test']
        batch_sizes = {'train': batch_size,'validation': 1, 'test':1 }

    else:
        parts = ['train', 'test']
        batch_sizes = {'train': batch_size, 'test':1 }
    

    standard_dataset = get_standard_dataset('lodopab', impl=IMPL)
    ray_trafo = standard_dataset.get_ray_trafo(impl=IMPL)
    
    CACHE_FILES = {part: (path.join(cache_dir, 'cache_lodopab_' + part + '_fbp.npy'), None) for part in parts }
    
    
    dataset = get_cached_fbp_dataset(standard_dataset, ray_trafo, CACHE_FILES)
        
   
    # create PyTorch datasets        
    datasets = {x: RandomAccessTorchDataset(dataset = dataset,
        part =  x, center_crop = dataset_image_center_crop, reshape=((1,) + dataset.space[0].shape,
                               (1,) + dataset.space[1].shape)) for x in parts}
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_sizes[x], pin_memory=True, num_workers = num_workers ) for x in parts}

    return dataloaders