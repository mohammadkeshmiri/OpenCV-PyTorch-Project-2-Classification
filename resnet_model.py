from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2 as transforms

class pretrained_resnet18:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_in, num_class)
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform



class pretrained_resnet50:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_in, num_class)
        self.name = "ResNet50"
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform
    
class pretrained_resnet101:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.ResNet101_Weights.DEFAULT
        self.model = models.resnet101(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_in, num_class)
        self.name = "ResNet101"
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform
    
class pretrained_resnet152:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.ResNet152_Weights.DEFAULT
        self.model = models.resnet152(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_in, num_class)
        self.name = "ResNet152"
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform


class pretrained_vgg16:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.VGG16_Weights.DEFAULT
        self.model = models.vgg16(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features=last_layer_in, out_features=num_class)
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform
    

class pretrained_vgg16_bn:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.VGG16_BN_Weights.DEFAULT
        self.model = models.vgg16_bn(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features=last_layer_in, out_features=num_class)
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform
    
class pretrained_vgg19_bn:
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, fix_feature_extractor=True, num_class=13):
        self.weights = models.VGG19_BN_Weights.DEFAULT
        self.model = models.vgg19_bn(self.weights)
        
        if fix_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
                
        last_layer_in = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features=last_layer_in, out_features=num_class)
    
    def get_model(self):
        return self.model
    
    def get_auto_transforms(self):
        return self.weights.transforms()
    
    def get_train_transforms(self):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)])

        return transform
    
    def get_test_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        return transform