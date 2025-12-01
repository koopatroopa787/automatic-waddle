"""
Configuration File
COMP64301: Computer Vision Coursework
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    """Base configuration class"""
    
    # Directory paths
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    AUGMENTED_DATA_DIR = DATA_DIR / 'augmented'
    
    MODELS_DIR = PROJECT_ROOT / 'models'
    CNN_MODELS_DIR = MODELS_DIR / 'cnn'
    TRAD_CV_MODELS_DIR = MODELS_DIR / 'traditional_cv'
    
    RESULTS_DIR = PROJECT_ROOT / 'results'
    CNN_RESULTS_DIR = RESULTS_DIR / 'cnn'
    TRAD_CV_RESULTS_DIR = RESULTS_DIR / 'traditional_cv'
    COMPARISON_DIR = RESULTS_DIR / 'comparison'
    
    # Dataset configuration
    DATASET_NAME = None  # To be set: 'cifar10', 'cifar100', 'custom', etc.
    NUM_CLASSES = None
    INPUT_SIZE = (224, 224)  # Default input size
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # CNN training configuration
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_PARAMS = {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_range': 15,
        'zoom_range': 0.1,
        'brightness_range': (0.8, 1.2),
    }
    
    # Traditional CV configuration
    FEATURE_DETECTOR = 'SIFT'  # Options: 'SIFT', 'SURF', 'ORB', 'Harris'
    DESCRIPTOR_TYPE = 'SIFT'  # Options: 'SIFT', 'SURF', 'ORB'
    VOCAB_SIZE = 500  # For Bag of Words
    CLASSIFIER = 'SVM'  # Options: 'SVM', 'RandomForest', 'KNN'
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Device configuration
    DEVICE = 'cuda'  # Will be set dynamically based on availability
    
    @classmethod
    def set_dataset(cls, dataset_name, num_classes, input_size=(224, 224)):
        """Set dataset-specific configuration"""
        cls.DATASET_NAME = dataset_name
        cls.NUM_CLASSES = num_classes
        cls.INPUT_SIZE = input_size
        print(f"Configuration updated for dataset: {dataset_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Input size: {input_size}")


class CIFAR10Config(Config):
    """CIFAR-10 specific configuration"""
    DATASET_NAME = 'cifar10'
    NUM_CLASSES = 10
    INPUT_SIZE = (32, 32)  # Native CIFAR-10 size
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR100Config(Config):
    """CIFAR-100 specific configuration"""
    DATASET_NAME = 'cifar100'
    NUM_CLASSES = 100
    INPUT_SIZE = (32, 32)  # Native CIFAR-100 size


class Caltech101Config(Config):
    """Caltech-101 specific configuration"""
    DATASET_NAME = 'caltech101'
    NUM_CLASSES = 101
    INPUT_SIZE = (224, 224)  # Higher resolution for Caltech-101


class CustomDatasetConfig(Config):
    """Custom dataset configuration template"""
    DATASET_NAME = 'custom'
    NUM_CLASSES = None  # To be set
    INPUT_SIZE = (224, 224)
