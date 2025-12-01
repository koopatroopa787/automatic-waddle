"""
Main Script for CNN Training on Caltech-101
COMP64301: Computer Vision Coursework

Run this script to train CNN models on Caltech-101
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from configs.config import Caltech101Config
from src.cnn.caltech101_loader import create_caltech101_loaders
from src.cnn.models import create_model
from src.cnn.train import train_model
from src.utils.helpers import set_seed, print_model_summary


def main():
    """
    Main function to run CNN training on Caltech-101
    """
    print("\n" + "="*80)
    print(" "*20 + "CALTECH-101 CNN TRAINING")
    print("="*80)
    
    # Configuration
    config = Caltech101Config()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    print("\n1. Loading Caltech-101 Dataset...")
    print("-" * 80)
    print("Note: First download may take a few minutes (~130 MB)")
    
    # Create data loaders
    data_loader = create_caltech101_loaders(
        data_dir=str(config.RAW_DATA_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=0.2,
        test_split=0.2,
        use_augmentation=config.USE_AUGMENTATION,
        input_size=config.INPUT_SIZE[0]
    )
    
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    print("\n2. Creating CNN Model...")
    print("-" * 80)
    print("Note: Using larger input size (224x224) for Caltech-101")
    
    # Create model - use improved or resnet18 for better performance on 101 classes
    model = create_model(
        model_name='improved',  # Options: 'baseline', 'improved', 'vgg', 'resnet18'
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.5,
        pretrained=False
    )
    
    # Print model summary
    print_model_summary(model)
    
    print("\n3. Starting Training...")
    print("-" * 80)
    print(f"Training on {config.NUM_CLASSES} classes")
    print("This may take longer than CIFAR-10 due to:")
    print("  - Higher resolution images (224x224 vs 32x32)")
    print("  - More classes (101 vs 10)")
    print("  - Smaller dataset per class")
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name='improved_cnn_caltech101'
    )
    
    print("\n" + "="*80)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*80)
    
    print("\nFinal Results:")
    print(f"  Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    
    print("\nResults saved to: results/cnn/")
    print("\nNext steps:")
    print("  1. Compare with CIFAR-10 results")
    print("  2. Visualize training curves")
    print("  3. Try different architectures (resnet18 recommended for 101 classes)")
    print("  4. Analyze per-class performance")
    

if __name__ == "__main__":
    main()
