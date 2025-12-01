"""
CNN Model Architectures
COMP64301: Computer Vision Coursework

This module contains various CNN architectures for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture for CIFAR-10
    Simple but effective architecture for benchmarking
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Initialize baseline CNN
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with residual connections
    More complex architecture for better performance
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        """
        Initialize improved CNN
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Block 1 (residual)
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # Block 2 (residual with downsampling)
        identity = self.downsample1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = F.relu(x + identity)
        
        # Block 3 (residual with downsampling)
        identity = self.downsample2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = F.relu(x + identity)
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class VGGStyleCNN(nn.Module):
    """
    VGG-style CNN architecture
    Stacked convolutional layers with batch normalization
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Initialize VGG-style CNN
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(VGGStyleCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet-18 architecture with optional pretrained weights
    Suitable for standard ImageNet-sized inputs (224x224)
    """
    
    def __init__(self, num_classes=10, pretrained=False):
        """
        Initialize ResNet-18
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ResNet18, self).__init__()
        
        # Load ResNet-18
        if pretrained:
            # For newer PyTorch versions (2.0+)
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except:
                # For older PyTorch versions
                self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=False)
        
        # Replace final fully connected layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class ResNet18Pretrained(nn.Module):
    """
    ResNet-18 with pretrained weights (transfer learning)
    Modified for CIFAR-10 input size
    """
    
    def __init__(self, num_classes=10, pretrained=True, freeze_layers=False):
        """
        Initialize pretrained ResNet-18
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_layers: Whether to freeze early layers
        """
        super(ResNet18Pretrained, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for CIFAR-10 (32x32 instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove max pooling (not needed for small images)
        self.resnet.maxpool = nn.Identity()
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Optionally freeze early layers
        if freeze_layers:
            self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze early layers for transfer learning"""
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.resnet(x)


def create_model(
    model_name='baseline',
    num_classes=10,
    dropout_rate=0.5,
    pretrained=False
):
    """
    Factory function to create CNN models
    
    Args:
        model_name: Name of the model ('baseline', 'improved', 'vgg', 'resnet18')
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights (for ResNet)
        
    Returns:
        CNN model instance
    """
    model_dict = {
        'baseline': BaselineCNN,
        'improved': ImprovedCNN,
        'vgg': VGGStyleCNN,
    }
    
    if model_name == 'resnet18':
        model = ResNet18Pretrained(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_name in model_dict:
        model = model_dict[model_name](
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN Models...")
    
    models_to_test = ['baseline', 'improved', 'vgg', 'resnet18']
    
    for model_name in models_to_test:
        print(f"\n{model_name.upper()} Model:")
        print("-" * 50)
        
        model = create_model(
            model_name=model_name,
            num_classes=10,
            dropout_rate=0.5,
            pretrained=False
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")