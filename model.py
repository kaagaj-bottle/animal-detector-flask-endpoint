from load_model import model
import torch


# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


def output_from_model(input):
    return model(input)
