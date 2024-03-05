import torch
import torchvision
from torchvision.models import efficientnet_b0

# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# transform required for images
auto_transforms = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()


# construct the vision model
def vision_model_constructor(output_shape: int = 55):

    vision_model = efficientnet_b0(weights=None).to(device)
    vision_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)).to(device)

    vision_model.load_state_dict(torch.load(
        "models_pth/model_march_3.pth", map_location=device))
    return vision_model


def audio_model_constructor(output_shape: int = 23):
    audio_model = efficientnet_b0(weights=None).to(device)
    audio_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)).to(device)

    audio_model.load_state_dict(torch.load(
        "models_pth/model_march_3.pth", map_location=device))
    return audio_model
