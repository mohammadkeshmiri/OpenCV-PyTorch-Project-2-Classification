import torch
import torchvision
from torchvision import transforms
from torchvision import models
from resnet_model import pretrained_resnet152


# Load the model
model = pretrained_resnet152()

auto_transforms = model.get_auto_transforms()
print(auto_transforms)