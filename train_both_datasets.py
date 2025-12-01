"""
Main Training Script for Multiple Datasets
COMP64301: Computer Vision Coursework

Train CNN models on both CIFAR-10 and CIFAR-100
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from configs.config import Config
from src.cnn.multi_dataset_loader import create_data_loaders
from src.cnn.models import create_model
from src.cnn.train import train_model
from src.utils.helpers import set_seed, print_model_summary


def train_on_dataset(dataset_name='cifar10', model_name='baseline', epochs=50):
    """
    Train a model on specified dataset
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        model_name: Model architecture name
        epochs: Number of training epochs
    """
    print("\n" + "="*80)
    print(f"Training {model_name.upper()} on {dataset_name.upper()}")
    print("="*80)
    
    # Configuration
    config = Config()
    config.DATASET_NAME = dataset_name
    config.EPOCHS = epochs
    
    # Set number of classes based on dataset
    if dataset_name == 'cifar10':
        config.NUM_CLASSES = 10
    elif dataset_name == 'cifar100':
        config.NUM_CLASSES = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    print("\n1. Loading Dataset...")
    print("-" * 80)
    
    # Create data loaders
    data_loader = create_data_loaders(
        dataset_name=dataset_name,
        data_dir=str(config.RAW_DATA_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=0.1,
        use_augmentation=config.USE_AUGMENTATION,
        input_size=config.INPUT_SIZE[0]
    )
    
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    print("\n2. Creating Model...")
    print("-" * 80)
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.5,
        pretrained=False
    )
    
    print_model_summary(model)
    
    print("\n3. Starting Training...")
    print("-" * 80)
    
    # Train model
    experiment_name = f'{model_name}_{dataset_name}'
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=experiment_name
    )
    
    print("\n" + "="*80)
    print(f"Training on {dataset_name.upper()} Complete!")
    print("="*80)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    
    return trained_model, history


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train CNN models on CIFAR-10 and/or CIFAR-100'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'both'],
        default='both',
        help='Which dataset(s) to train on'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='baseline',
        choices=['baseline', 'improved', 'vgg', 'resnet18'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" "*15 + "MULTI-DATASET CNN TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset(s): {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    
    results = {}
    
    if args.dataset in ['cifar10', 'both']:
        model, history = train_on_dataset(
            dataset_name='cifar10',
            model_name=args.model,
            epochs=args.epochs
        )
        results['cifar10'] = {
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1]
        }
    
    if args.dataset in ['cifar100', 'both']:
        model, history = train_on_dataset(
            dataset_name='cifar100',
            model_name=args.model,
            epochs=args.epochs
        )
        results['cifar100'] = {
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1]
        }
    
    # Print summary
    print("\n" + "="*80)
    print(" "*20 + "TRAINING SUMMARY")
    print("="*80)
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        print(f"  Final Training Accuracy: {result['final_train_acc']:.2f}%")
    
    print("\n" + "="*80)
    print("All training complete!")
    print("Results saved in: results/cnn/")
    print("="*80)


if __name__ == "__main__":
    main()
