"""
Traditional Computer Vision - Bag of Visual Words
COMP64301: Computer Vision Coursework

Implements Bag-of-Words representation using k-means clustering
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from pathlib import Path
from tqdm import tqdm


class BagOfWords:
    """
    Bag-of-Words model for image representation
    """
    
    def __init__(self, vocab_size=500, random_state=42):
        """
        Initialize Bag-of-Words model
        
        Args:
            vocab_size: Number of visual words (cluster centers)
            random_state: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.kmeans = None
        self.vocabulary = None
        
        print(f"Initialized Bag-of-Words model")
        print(f"  Vocabulary size: {vocab_size}")
    
    def build_vocabulary(self, all_descriptors, max_samples=None):
        """
        Build visual vocabulary using k-means clustering
        
        Args:
            all_descriptors: List of descriptor arrays from all images
            max_samples: Maximum number of descriptors to use for clustering
        """
        print("\nBuilding visual vocabulary...")
        
        # Concatenate all descriptors
        print("Concatenating descriptors...")
        all_desc_concat = np.vstack([d for d in all_descriptors if len(d) > 0])
        
        print(f"Total descriptors: {len(all_desc_concat):,}")
        
        # Sample if too many
        if max_samples is not None and len(all_desc_concat) > max_samples:
            print(f"Sampling {max_samples:,} descriptors for clustering...")
            indices = np.random.choice(len(all_desc_concat), max_samples, replace=False)
            all_desc_concat = all_desc_concat[indices]
        
        # Perform k-means clustering
        print(f"Performing k-means clustering (k={self.vocab_size})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            random_state=self.random_state,
            batch_size=1000,
            max_iter=100,
            verbose=1
        )
        
        self.kmeans.fit(all_desc_concat)
        self.vocabulary = self.kmeans.cluster_centers_
        
        print(f"✓ Vocabulary built with {self.vocab_size} visual words")
        print(f"  Descriptor dimension: {self.vocabulary.shape[1]}")
    
    def encode_image(self, descriptors):
        """
        Encode image descriptors as histogram over visual words
        
        Args:
            descriptors: Descriptor array for single image (n_features, descriptor_dim)
            
        Returns:
            histogram: numpy array of shape (vocab_size,)
        """
        if len(descriptors) == 0:
            # No features, return zero histogram
            return np.zeros(self.vocab_size)
        
        # Assign each descriptor to nearest cluster center
        labels = self.kmeans.predict(descriptors)
        
        # Create histogram
        histogram, _ = np.histogram(labels, bins=np.arange(self.vocab_size + 1))
        
        # Normalize
        if histogram.sum() > 0:
            histogram = histogram.astype(np.float32) / histogram.sum()
        
        return histogram
    
    def encode_dataset(self, all_descriptors):
        """
        Encode all images in dataset
        
        Args:
            all_descriptors: List of descriptor arrays
            
        Returns:
            features: numpy array of shape (n_images, vocab_size)
        """
        print(f"\nEncoding {len(all_descriptors)} images...")
        
        features = []
        for descriptors in tqdm(all_descriptors):
            histogram = self.encode_image(descriptors)
            features.append(histogram)
        
        features = np.array(features)
        
        print(f"✓ Encoded features shape: {features.shape}")
        
        return features
    
    def save_model(self, save_path):
        """Save Bag-of-Words model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'vocabulary': self.vocabulary,
            'kmeans': self.kmeans
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to: {save_path}")
    
    @staticmethod
    def load_model(load_path):
        """Load saved Bag-of-Words model"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        bow = BagOfWords(vocab_size=data['vocab_size'])
        bow.vocabulary = data['vocabulary']
        bow.kmeans = data['kmeans']
        
        print(f"Loaded Bag-of-Words model")
        print(f"  Vocabulary size: {bow.vocab_size}")
        
        return bow


if __name__ == "__main__":
    print("Testing Bag-of-Words...")
    
    # Create some random descriptors for testing
    print("\nGenerating test data...")
    n_images = 100
    n_features_per_image = 50
    descriptor_dim = 128  # SIFT
    
    all_descriptors = [
        np.random.randn(n_features_per_image, descriptor_dim).astype(np.float32)
        for _ in range(n_images)
    ]
    
    print(f"Created {n_images} images with {n_features_per_image} features each")
    
    # Build vocabulary
    print("\n" + "="*80)
    bow = BagOfWords(vocab_size=50)
    bow.build_vocabulary(all_descriptors, max_samples=1000)
    
    # Encode images
    print("\n" + "="*80)
    features = bow.encode_dataset(all_descriptors)
    
    print(f"\nResults:")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Sparsity: {(features == 0).sum() / features.size * 100:.1f}% zeros")
    
    print("\n✓ Bag-of-Words test complete!")
