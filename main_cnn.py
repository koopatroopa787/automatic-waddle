"""
Main Script for CNN Training
COMP64301: Computer Vision Coursework

Run this script to train CNN models on CIFAR-10
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from configs.config import CIFAR10Config
from src.cnn.data_loader import create_cifar10_loaders
from src.cnn.models import create_model
from src.cnn.train import train_model
from src.utils.helpers import set_seed, print_model_summary


def main():
    """
    Main function to run CNN training
    """
    print("\n" + "="*80)
    print(" "*20 + "CIFAR-10 CNN TRAINING")
    print("="*80)
    
    # Configuration
    config = CIFAR10Config()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    print("\n1. Loading CIFAR-10 Dataset...")
    print("-" * 80)
    
    # Create data loaders
    data_loader = create_cifar10_loaders(
        data_dir=str(config.RAW_DATA_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=0.1,
        use_augmentation=config.USE_AUGMENTATION,
        input_size=config.INPUT_SIZE[0]
    )
    
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    print("\n2. Creating CNN Model...")
    print("-" * 80)
    
    # Create model
    model = create_model(
        model_name='baseline',  # Options: 'baseline', 'improved', 'vgg', 'resnet18'
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.5,
        pretrained=False
    )
    
    # Print model summary
    print_model_summary(model)
    
    print("\n3. Starting Training...")
    print("-" * 80)
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name='baseline_cnn_cifar10'
    )
    
    print("\n" + "="*80)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*80)
    
    print("\nFinal Results:")
    print(f"  Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    
    print("\nResults saved to: results/cnn/")
    print("\nNext steps:")
    print("  1. Visualize training curves")
    print("  2. Try hyperparameter tuning")
    print("  3. Experiment with different architectures")
    

if __name__ == "__main__":
    main()
