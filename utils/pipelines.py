import numpy as np
import torch

def random(shape, seed=0):
    np.random.seed(seed)
    x = np.random.normal(size=shape)
    return torch.tensor(x, dtype=torch.float32)
