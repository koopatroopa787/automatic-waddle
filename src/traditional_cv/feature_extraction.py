"""
Traditional Computer Vision - Feature Extraction
COMP64301: Computer Vision Coursework

Implements SIFT and ORB feature extraction for Bag-of-Words
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import torch
from torchvision import datasets, transforms


class FeatureExtractor:
    """
    Extract local features (SIFT or ORB) from images
    """
    
    def __init__(self, feature_type='SIFT', max_features=500):
        """
        Initialize feature extractor
        
        Args:
            feature_type: 'SIFT' or 'ORB'
            max_features: Maximum number of features per image
        """
        self.feature_type = feature_type.upper()
        self.max_features = max_features
        
        # Create detector
        if self.feature_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif self.feature_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        print(f"Initialized {self.feature_type} feature extractor")
        print(f"  Max features per image: {max_features}")
    
    def extract_from_image(self, image):
        """
        Extract features from a single image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            descriptors: numpy array of shape (n_features, descriptor_dim)
        """
        # Convert PIL to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            # No features found, return empty array
            if self.feature_type == 'SIFT':
                return np.array([]).reshape(0, 128)
            else:  # ORB
                return np.array([]).reshape(0, 32)
        
        return descriptors.astype(np.float32)
    
    def extract_from_dataset(self, dataset, max_images=None):
        """
        Extract features from entire dataset
        
        Args:
            dataset: PyTorch dataset or list of images
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            all_descriptors: List of descriptor arrays
            labels: List of labels
        """
        all_descriptors = []
        labels = []
        
        n_images = len(dataset) if max_images is None else min(max_images, len(dataset))
        
        print(f"\nExtracting {self.feature_type} features from {n_images} images...")
        
        for i in tqdm(range(n_images)):
            # Get image and label
            if isinstance(dataset, torch.utils.data.Dataset):
                image, label = dataset[i]
            else:
                image, label = dataset[i]
            
            # Extract features
            descriptors = self.extract_from_image(image)
            
            if len(descriptors) > 0:
                all_descriptors.append(descriptors)
                labels.append(label)
        
        print(f"Extracted features from {len(all_descriptors)} images")
        print(f"Total features: {sum(len(d) for d in all_descriptors)}")
        
        return all_descriptors, labels
    
    def save_features(self, descriptors, labels, save_path):
        """Save extracted features"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'descriptors': descriptors,
            'labels': labels,
            'feature_type': self.feature_type,
            'max_features': self.max_features
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Features saved to: {save_path}")
    
    @staticmethod
    def load_features(load_path):
        """Load saved features"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {data['feature_type']} features")
        print(f"  Number of images: {len(data['descriptors'])}")
        print(f"  Total features: {sum(len(d) for d in data['descriptors'])}")
        
        return data['descriptors'], data['labels']


def create_cifar10_raw_dataset(data_dir='./data/raw'):
    """
    Create CIFAR-10 dataset without normalization (for feature extraction)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte().numpy().transpose(1, 2, 0))
    ])
    
    # Load train and test sets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def create_caltech101_raw_dataset(data_dir='./data/raw'):
    """
    Create Caltech-101 dataset without normalization (for feature extraction)
    """
    from torch.utils.data import Subset
    
    # Load full dataset
    full_dataset = datasets.Caltech101(
        root=data_dir,
        download=True,
        transform=None
    )
    
    # Remove background class and convert to numpy
    def transform_fn(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    class RawCaltech101:
        def __init__(self, dataset):
            # Filter out background class
            self.indices = [i for i, (_, label) in enumerate(dataset) if label != 0]
            self.dataset = dataset
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.dataset[real_idx]
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return np.array(img), label - 1  # Shift labels
    
    return RawCaltech101(full_dataset)


if __name__ == "__main__":
    print("Testing Feature Extraction...")
    
    # Test SIFT
    print("\n" + "="*80)
    print("Testing SIFT Feature Extraction")
    print("="*80)
    
    sift = FeatureExtractor('SIFT', max_features=100)
    
    # Load small sample from CIFAR-10
    train_data, _ = create_cifar10_raw_dataset()
    
    # Extract from 100 images
    descriptors, labels = sift.extract_from_dataset(train_data, max_images=100)
    
    print(f"\nSample results:")
    print(f"  Images processed: {len(descriptors)}")
    print(f"  Features per image (avg): {np.mean([len(d) for d in descriptors]):.1f}")
    print(f"  Descriptor dimension: {descriptors[0].shape[1] if len(descriptors) > 0 else 0}")
    
    # Test ORB
    print("\n" + "="*80)
    print("Testing ORB Feature Extraction")
    print("="*80)
    
    orb = FeatureExtractor('ORB', max_features=100)
    descriptors_orb, labels_orb = orb.extract_from_dataset(train_data, max_images=100)
    
    print(f"\nSample results:")
    print(f"  Images processed: {len(descriptors_orb)}")
    print(f"  Features per image (avg): {np.mean([len(d) for d in descriptors_orb]):.1f}")
    print(f"  Descriptor dimension: {descriptors_orb[0].shape[1] if len(descriptors_orb) > 0 else 0}")
    
    print("\n✓ Feature extraction test complete!")
