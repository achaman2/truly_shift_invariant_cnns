import torch
import torch.nn as nn
import numpy as np

def compute_num_params(model):

    num_p = 0

    for params in model.parameters():
        num_p = num_p + np.prod(params.shape)

    print('Number of parameters: ', num_p)

    return num_p


