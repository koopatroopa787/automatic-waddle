"""
Multi-Dataset Data Loader
COMP64301: Computer Vision Coursework

Support for both CIFAR-10 and CIFAR-100 datasets.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class MultiDatasetLoader:
    """
    Unified data loader for CIFAR-10 and CIFAR-100
    """
    
    DATASET_INFO = {
        'cifar10': {
            'class': datasets.CIFAR10,
            'num_classes': 10,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'class_names': [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        },
        'cifar100': {
            'class': datasets.CIFAR100,
            'num_classes': 100,
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'class_names': None  # Will be loaded from dataset
        }
    }
    
    def __init__(
        self,
        dataset_name='cifar10',
        data_dir='./data/raw',
        batch_size=32,
        num_workers=4,
        validation_split=0.1,
        use_augmentation=True,
        input_size=32,
        download=True
    ):
        """
        Initialize multi-dataset loader
        
        Args:
            dataset_name: 'cifar10' or 'cifar100'
            data_dir: Directory to store/load data
            batch_size: Batch size for training
            num_workers: Number of worker processes
            validation_split: Fraction for validation
            use_augmentation: Whether to apply augmentation
            input_size: Input image size
            download: Whether to download if not present
        """
        if dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Dataset must be 'cifar10' or 'cifar100', got: {dataset_name}")
        
        self.dataset_name = dataset_name
        self.dataset_info = self.DATASET_INFO[dataset_name]
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
        self._load_class_names()
    
    def _setup_transforms(self):
        """Setup data transformations"""
        mean = self.dataset_info['mean']
        std = self.dataset_info['std']
        
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        if self.input_size != 32:
            base_transform.insert(0, transforms.Resize(self.input_size))
        
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
        
        self.test_transform = transforms.Compose(base_transform)
        
        print(f"\n{self.dataset_name.upper()} Data transformations configured:")
        print(f"  - Input size: {self.input_size}x{self.input_size}")
        print(f"  - Augmentation: {self.use_augmentation}")
        print(f"  - Normalization: mean={mean}, std={std}")
    
    def _load_datasets(self):
        """Load datasets"""
        print(f"\nLoading {self.dataset_name.upper()} dataset...")
        
        dataset_class = self.dataset_info['class']
        
        # Load training dataset
        train_dataset_full = dataset_class(
            root=self.data_dir,
            train=True,
            transform=self.train_transform,
            download=self.download
        )
        
        # Split into train and validation
        if self.validation_split > 0:
            train_size = int((1 - self.validation_split) * len(train_dataset_full))
            val_size = len(train_dataset_full) - train_size
            
            train_dataset, val_dataset = random_split(
                train_dataset_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Validation uses test transforms
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
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Load test dataset
        test_dataset = dataset_class(
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
        print(f"Dataset loading complete!\n")
    
    def _load_class_names(self):
        """Load class names for the dataset"""
        if self.dataset_info['class_names'] is None:
            # For CIFAR-100, load from dataset
            if self.dataset_name == 'cifar100':
                # Get from the actual dataset
                test_dataset = self.test_loader.dataset
                if hasattr(test_dataset, 'classes'):
                    self.class_names = test_dataset.classes
                else:
                    # Fallback: load fresh dataset just to get class names
                    temp_dataset = datasets.CIFAR100(
                        root=self.data_dir,
                        train=False,
                        download=False
                    )
                    self.class_names = temp_dataset.classes
        else:
            self.class_names = self.dataset_info['class_names']
    
    def get_loaders(self):
        """Get data loaders"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_num_classes(self):
        """Get number of classes"""
        return self.dataset_info['num_classes']
    
    def get_class_names(self):
        """Get class names"""
        return self.class_names
    
    def visualize_samples(self, num_samples=16, save_path=None):
        """Visualize random samples"""
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        images = self._denormalize(images)
        
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(num_samples):
            if idx < len(images):
                img = images[idx].numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                axes[idx].imshow(img)
                axes[idx].set_title(
                    self.class_names[labels[idx]],
                    fontsize=8
                )
                axes[idx].axis('off')
            else:
                axes[idx].axis('off')
        
        plt.suptitle(f'{self.dataset_name.upper()} Dataset Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def _denormalize(self, tensor):
        """Denormalize tensor for visualization"""
        mean = torch.tensor(self.dataset_info['mean']).view(3, 1, 1)
        std = torch.tensor(self.dataset_info['std']).view(3, 1, 1)
        return tensor * std + mean


def create_data_loaders(
    dataset_name='cifar10',
    data_dir='./data/raw',
    batch_size=32,
    num_workers=4,
    validation_split=0.1,
    use_augmentation=True,
    input_size=32
):
    """
    Convenience function to create data loaders
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of workers
        validation_split: Validation fraction
        use_augmentation: Whether to augment
        input_size: Input image size
        
    Returns:
        MultiDatasetLoader instance
    """
    loader = MultiDatasetLoader(
        dataset_name=dataset_name,
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
    print("Testing Multi-Dataset Loader...")
    
    for dataset in ['cifar10', 'cifar100']:
        print(f"\n{'='*80}")
        print(f"Testing {dataset.upper()}")
        print('='*80)
        
        loader = create_data_loaders(
            dataset_name=dataset,
            data_dir='../../data/raw',
            batch_size=32,
            validation_split=0.1
        )
        
        train_loader, val_loader, test_loader = loader.get_loaders()
        
        print(f"\nDataset: {dataset.upper()}")
        print(f"Number of classes: {loader.get_num_classes()}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test batch shape
        images, labels = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
