import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

class UnifiedDataLoader:
    def __init__(self, config, dataset_name, use_augmentation=True):
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.use_augmentation = use_augmentation
        self.data_dir = Path(config.RAW_DATA_DIR)
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._setup_transforms()
        self._load_dataset()

    def _setup_transforms(self):
        # Default to ImageNet stats (works well for both)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        input_size = self.config.INPUT_SIZE[0]

        base = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        if self.use_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ] + base[1:]) # Skip resize in base for train
        else:
            self.train_transform = transforms.Compose(base)

        self.test_transform = transforms.Compose(base)

    def _load_dataset(self):
        print(f"\nLoading {self.dataset_name.upper()}...")
        
        if self.dataset_name == 'cifar10':
            full_train = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transform)
            test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.test_transform)
            
            # Split train/val (90/10)
            train_size = int(0.9 * len(full_train))
            val_size = len(full_train) - train_size
            train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
            # Force val dataset to use test transforms (no augmentation)
            val_dataset.dataset.transform = self.test_transform 

        elif self.dataset_name == 'caltech101':
            full_dataset = datasets.Caltech101(root=self.data_dir, download=True)
            # Filter background class
            indices = [i for i, (_, label) in enumerate(full_dataset) if label != 0]
            
            # Custom wrapper to handle label shift and transforms
            class CaltechWrapper(torch.utils.data.Dataset):
                def __init__(self, indices, dataset, transform):
                    self.indices = indices
                    self.dataset = dataset
                    self.transform = transform
                def __len__(self): return len(self.indices)
                def __getitem__(self, idx):
                    img, label = self.dataset[self.indices[idx]]
                    if img.mode != 'RGB': img = img.convert('RGB')
                    if self.transform: img = self.transform(img)
                    return img, label - 1 # Shift labels down

            # Split indices (60/20/20)
            np.random.shuffle(indices)
            train_idx = indices[:int(0.6 * len(indices))]
            val_idx = indices[int(0.6 * len(indices)):int(0.8 * len(indices))]
            test_idx = indices[int(0.8 * len(indices)):]

            train_dataset = CaltechWrapper(train_idx, full_dataset, self.train_transform)
            val_dataset = CaltechWrapper(val_idx, full_dataset, self.test_transform)
            test_dataset = CaltechWrapper(test_idx, full_dataset, self.test_transform)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader