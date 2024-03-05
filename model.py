from load_model import vision_model_constructor, audio_model_constructor
import torch

vision_model = vision_model_constructor()
audio_model = audio_model_constructor()


# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


def vision_output(input):
    vision_model.eval()
    with torch.no_grad():
        return vision_model(input)


def audio_output(input):
    audio_model.eval()
    with torch.no_grad():
        return audio_model(input)
