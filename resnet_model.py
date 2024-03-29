from torchvision import models
import torch.nn as nn

def pretrained_resnet18(transfer_learning=True, num_class=13):
    resnet = models.resnet18(pretrained=True)
    
    if transfer_learning:
        for param in resnet.parameters():
            param.requires_grad = False
            
    last_layer_in = resnet.fc.in_features
    resnet.fc = nn.Linear(last_layer_in, num_class)
    
    return resnet