"""
CIFAR-10 Data Loader
COMP64301: Computer Vision Coursework

This module handles downloading, loading, and preprocessing CIFAR-10 dataset.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class CIFAR10DataLoader:
    """
    CIFAR-10 Dataset Loader with preprocessing and augmentation
    """
    
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        validation_split=0.1,
        use_augmentation=True,
        input_size=32,
        download=True
    ):
        """
        Initialize CIFAR-10 data loader
        
        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            validation_split: Fraction of training data to use for validation
            use_augmentation: Whether to apply data augmentation
            input_size: Input image size (can upscale from native 32x32)
            download: Whether to download dataset if not present
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.use_augmentation = use_augmentation
        self.input_size = input_size
        self.download = download
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._setup_transforms()
        self._load_datasets()
    
    def _setup_transforms(self):
        """Setup data transformations for training and testing"""
        
        # Base transforms for all data
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
        ]
        
        # Add resize if input_size is different from native 32x32
        if self.input_size != 32:
            base_transform.insert(0, transforms.Resize(self.input_size))
        
        # Training transforms with augmentation
        if self.use_augmentation:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.RandomRotation(15),
            ] + base_transform)
        else:
            self.train_transform = transforms.Compose(base_transform)
        
        # Test/validation transforms (no augmentation)
        self.test_transform = transforms.Compose(base_transform)
        
        print("Data transformations configured:")
        print(f"  - Input size: {self.input_size}x{self.input_size}")
        print(f"  - Augmentation: {self.use_augmentation}")
    
    def _load_datasets(self):
        """Load CIFAR-10 datasets"""
        
        print("\nLoading CIFAR-10 dataset...")
        
        # Load full training dataset
        train_dataset_full = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.train_transform,
            download=self.download
        )
        
        # Split training data into train and validation
        if self.validation_split > 0:
            train_size = int((1 - self.validation_split) * len(train_dataset_full))
            val_size = len(train_dataset_full) - train_size
            
            train_dataset, val_dataset = random_split(
                train_dataset_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create validation loader with test transforms (no augmentation)
            val_dataset.dataset.transform = self.test_transform
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            print(f"  - Training samples: {train_size}")
            print(f"  - Validation samples: {val_size}")
        else:
            train_dataset = train_dataset_full
            print(f"  - Training samples: {len(train_dataset)}")
        
        # Create training loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Load test dataset
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.test_transform,
            download=self.download
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"  - Test samples: {len(test_dataset)}")
        print("Dataset loading complete!\n")
    
    def get_loaders(self):
        """
        Get data loaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def visualize_samples(self, num_samples=16, save_path=None):
        """
        Visualize random samples from training data
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        # Get a batch of training data
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Denormalize images for visualization
        images = self._denormalize(images)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(num_samples):
            if idx < len(images):
                img = images[idx].numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                axes[idx].imshow(img)
                axes[idx].set_title(
                    self.CLASS_NAMES[labels[idx]],
                    fontsize=10
                )
                axes[idx].axis('off')
            else:
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def _denormalize(self, tensor):
        """
        Denormalize tensor for visualization
        
        Args:
            tensor: Normalized image tensor
            
        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(self.CIFAR10_MEAN).view(3, 1, 1)
        std = torch.tensor(self.CIFAR10_STD).view(3, 1, 1)
        return tensor * std + mean
    
    def get_class_distribution(self):
        """
        Get class distribution in training data
        
        Returns:
            dict: Class counts
        """
        train_dataset = self.train_loader.dataset
        
        if hasattr(train_dataset, 'dataset'):
            labels = [train_dataset.dataset.targets[i] 
                     for i in train_dataset.indices]
        else:
            labels = train_dataset.targets
        
        class_counts = {}
        for i, class_name in enumerate(self.CLASS_NAMES):
            class_counts[class_name] = labels.count(i)
        
        return class_counts


def create_cifar10_loaders(
    data_dir='./data/raw',
    batch_size=32,
    num_workers=4,
    validation_split=0.1,
    use_augmentation=True,
    input_size=32
):
    """
    Convenience function to create CIFAR-10 data loaders
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        validation_split: Fraction for validation
        use_augmentation: Whether to use data augmentation
        input_size: Input image size
        
    Returns:
        CIFAR10DataLoader instance
    """
    loader = CIFAR10DataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
        use_augmentation=use_augmentation,
        input_size=input_size,
        download=True
    )
    
    return loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing CIFAR-10 Data Loader...")
    
    loader = create_cifar10_loaders(
        data_dir='../../data/raw',
        batch_size=32,
        validation_split=0.1,
        use_augmentation=True
    )
    
    train_loader, val_loader, test_loader = loader.get_loaders()
    
    print("\nData loader statistics:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\nClass distribution:")
    class_dist = loader.get_class_distribution()
    for class_name, count in class_dist.items():
        print(f"  {class_name}: {count}")
    
    print("\nSample batch shape:")
    images, labels = next(iter(train_loader))
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
