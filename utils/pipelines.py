import numpy as np
import torch

def get_random(shape, seed=0):
    np.random.seed(seed)
    x = np.random.normal(size=shape)
    return torch.tensor(x, dtype=torch.float32)

def init_xavier_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)