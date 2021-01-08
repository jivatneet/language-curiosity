import math
import numpy as np
import torch

def normal(x, mu, sigma, device):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()

    pi = torch.Tensor(pi).to(device)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b