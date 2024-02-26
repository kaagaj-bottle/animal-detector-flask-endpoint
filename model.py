import torch
import torchvision
from torch import nn
from torchvision import transforms


# check the device
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

weights=torc
