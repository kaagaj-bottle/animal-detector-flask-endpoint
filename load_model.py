import torch
import torchvision
from torchvision.models import efficientnet_b0

# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# transform required for images
auto_transforms = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()


# construct the model
model = efficientnet_b0(weights=None).to(device)
output_shape = 55
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)).to(device)


model.load_state_dict(torch.load("model_cpu_state_dict.pth"))

