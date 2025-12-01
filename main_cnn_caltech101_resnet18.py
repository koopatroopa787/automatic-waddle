"""
Standalone Improved CNN Training on Caltech-101
COMP64301: Computer Vision Coursework

Self-contained script with ResNet-18 and all training logic included
Target: 60-70% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import json
from pathlib import Path
import numpy as np


# ============================================================================
# Model Definition
# ============================================================================

class ResNet18Caltech(nn.Module):
    """ResNet-18 for Caltech-101 with pretrained weights"""
    
    def __init__(self, num_classes=101, pretrained=True):
        super(ResNet18Caltech, self).__init__()
        
        # Load ResNet-18
        if pretrained:
            print("  Loading pretrained ImageNet weights...")
            try:
                # PyTorch 2.0+
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except:
                # Older PyTorch
                self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=False)
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Data Loading
# ============================================================================

def get_transforms():
    """Get training and validation transforms"""
    
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def create_data_loaders(batch_size=32, num_workers=0):
    """Create train and validation data loaders"""
    
    print("\n" + "-"*80)
    print("Loading Caltech-101 Dataset")
    print("-"*80)
    
    train_transforms, val_transforms = get_transforms()
    
    # Load full dataset
    full_dataset = datasets.Caltech101(
        root='./data/raw',
        download=True,
        transform=None
    )
    
    # Remove background class (class 0) and split
    non_background_indices = [i for i, (_, label) in enumerate(full_dataset) if label != 0]
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(non_background_indices)
    
    split_idx = int(0.8 * len(non_background_indices))
    train_indices = non_background_indices[:split_idx]
    val_indices = non_background_indices[split_idx:]
    
    # Custom dataset with transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.dataset[real_idx]
            
            # Convert grayscale to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label - 1  # Shift labels (remove background class)
    
    train_dataset = TransformedSubset(full_dataset, train_indices, train_transforms)
    val_dataset = TransformedSubset(full_dataset, val_indices, val_transforms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" "*15 + "IMPROVED CALTECH-101 CNN TRAINING")
    print(" "*20 + "(Target: 60-70% Accuracy)")
    print("="*80)
    
    # Configuration
    config = {
        'num_classes': 101,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stopping_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0
    }
    
    print("\n" + "-"*80)
    print("Configuration")
    print("-"*80)
    print(f"  Model: ResNet-18 (pretrained)")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Max epochs: {config['epochs']}")
    print(f"  Device: {config['device']}")
    
    if config['device'] == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\n" + "-"*80)
    print("Creating Model")
    print("-"*80)
    
    model = ResNet18Caltech(num_classes=config['num_classes'], pretrained=True)
    model = model.to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    print("✓ Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
    print("✓ Scheduler: ReduceLROnPlateau (patience=5)")
    
    # Training
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    save_dir = Path('results/cnn/resnet18_caltech101_improved')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config['device']
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{config['epochs']}] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}%", end='')
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pth')
            
            print(" ✓ NEW BEST!")
        else:
            patience_counter += 1
            print()
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config['early_stopping_patience']} epochs)")
            break
    
    training_time = (time.time() - start_time) / 60
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Final results
    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*80)
    
    print("\n" + "-"*80)
    print("Final Results")
    print("-"*80)
    print(f"✓ Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"✓ Best Epoch: {best_epoch}")
    print(f"✓ Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"✓ Training Time: {training_time:.1f} minutes")
    print(f"✓ Total Epochs: {len(history['train_loss'])}")
    
    # Save final results
    results = {
        'model': 'ResNet-18',
        'dataset': 'Caltech-101',
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
        'final_train_acc': float(history['train_acc'][-1]),
        'training_time_minutes': float(training_time),
        'total_epochs': len(history['train_loss']),
        'config': config
    }
    
    with open(save_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {save_dir}")
    
    # Comparison with previous
    print("\n" + "="*80)
    print("Improvement Analysis")
    print("="*80)
    
    previous_acc = 46.42
    improvement = best_val_acc - previous_acc
    
    print(f"\n📊 Performance Comparison:")
    print(f"  Previous Model: {previous_acc:.2f}%")
    print(f"  Improved Model: {best_val_acc:.2f}%")
    print(f"  Improvement: +{improvement:.2f}%")
    
    if best_val_acc >= 60:
        print("\n🎉 Target achieved! (60%+ accuracy)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()