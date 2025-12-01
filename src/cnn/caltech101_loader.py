"""
Caltech-101 Data Loader
COMP64301: Computer Vision Coursework

This module handles downloading, loading, and preprocessing Caltech-101 dataset.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class Caltech101DataLoader:
    """
    Caltech-101 Dataset Loader with preprocessing and augmentation
    """
    
    # ImageNet normalization (since Caltech-101 is similar to natural images)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        validation_split=0.2,
        test_split=0.2,
        use_augmentation=True,
        input_size=224,
        download=True
    ):
        """
        Initialize Caltech-101 data loader
        
        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            use_augmentation: Whether to apply data augmentation
            input_size: Input image size (Caltech-101 has variable sizes)
            download: Whether to download dataset if not present
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.test_split = test_split
        self.use_augmentation = use_augmentation
        self.input_size = input_size
        self.download = download
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_classes = None
        
        self._setup_transforms()
        self._load_datasets()
    
    def _setup_transforms(self):
        """Setup data transformations for training and testing"""
        
        # Base transforms for all data
        base_transform = [
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ]
        
        # Training transforms with augmentation
        if self.use_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((self.input_size + 32, self.input_size + 32)),
                transforms.RandomCrop(self.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
            ])
        else:
            self.train_transform = transforms.Compose(base_transform)
        
        # Test/validation transforms (no augmentation)
        self.test_transform = transforms.Compose(base_transform)
        
        print("Data transformations configured:")
        print(f"  - Input size: {self.input_size}x{self.input_size}")
        print(f"  - Augmentation: {self.use_augmentation}")
    
    def _load_datasets(self):
        """Load Caltech-101 dataset"""
        
        print("\nLoading Caltech-101 dataset...")
        print("Note: First download may take a few minutes (~130 MB)")
        
        # Load full dataset
        full_dataset = datasets.Caltech101(
            root=self.data_dir,
            transform=None,  # We'll set this later
            download=self.download
        )
        
        # Remove background class (index 0) - it's not a real object class
        # This is standard practice for Caltech-101
        indices = [i for i, (_, label) in enumerate(full_dataset) if label != 0]
        
        # Create subset without background class
        from torch.utils.data import Subset
        dataset_no_bg = Subset(full_dataset, indices)
        
        # Adjust labels (shift down by 1 since we removed class 0)
        class CustomSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
            
            def __getitem__(self, idx):
                img, label = self.subset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label - 1  # Shift labels
            
            def __len__(self):
                return len(self.subset)
        
        dataset = CustomSubset(dataset_no_bg)
        
        # Caltech-101 has 101 classes (excluding background)
        self.num_classes = 101
        
        # Split into train, validation, and test
        total_size = len(dataset)
        test_size = int(self.test_split * total_size)
        val_size = int(self.validation_split * total_size)
        train_size = total_size - test_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply transforms
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.test_transform
        test_dataset.dataset.transform = self.test_transform
        
        print(f"  - Training samples: {train_size}")
        print(f"  - Validation samples: {val_size}")
        print(f"  - Test samples: {test_size}")
        print(f"  - Number of classes: {self.num_classes}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
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
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(num_samples):
            if idx < len(images):
                img = images[idx].numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                axes[idx].imshow(img)
                axes[idx].set_title(f'Class {labels[idx].item()}', fontsize=10)
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
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        return tensor * std + mean


def create_caltech101_loaders(
    data_dir='./data/raw',
    batch_size=32,
    num_workers=4,
    validation_split=0.2,
    test_split=0.2,
    use_augmentation=True,
    input_size=224
):
    """
    Convenience function to create Caltech-101 data loaders
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        validation_split: Fraction for validation
        test_split: Fraction for testing
        use_augmentation: Whether to use data augmentation
        input_size: Input image size
        
    Returns:
        Caltech101DataLoader instance
    """
    loader = Caltech101DataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
        test_split=test_split,
        use_augmentation=use_augmentation,
        input_size=input_size,
        download=True
    )
    
    return loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing Caltech-101 Data Loader...")
    
    loader = create_caltech101_loaders(
        data_dir='../../data/raw',
        batch_size=32,
        validation_split=0.2,
        test_split=0.2,
        use_augmentation=True,
        input_size=224
    )
    
    train_loader, val_loader, test_loader = loader.get_loaders()
    
    print("\nData loader statistics:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Number of classes: {loader.num_classes}")
    
    print("\nSample batch shape:")
    images, labels = next(iter(train_loader))
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
