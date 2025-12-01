"""
Unified Training Script for Multiple Datasets
COMP64301: Computer Vision Coursework

Train CNN models on both CIFAR-10 and Caltech-101 for comprehensive evaluation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from configs.config import CIFAR10Config, Caltech101Config
from src.cnn.data_loader import create_cifar10_loaders
from src.cnn.caltech101_loader import create_caltech101_loaders
from src.cnn.models import create_model
from src.cnn.train import train_model
from src.utils.helpers import set_seed, print_model_summary
import json


def train_on_dataset(dataset_name, model_name='baseline'):
    """
    Train a model on a specific dataset
    
    Args:
        dataset_name: 'cifar10' or 'caltech101'
        model_name: Name of model architecture
    
    Returns:
        Training history
    """
    print("\n" + "="*80)
    print(f" "*25 + f"TRAINING ON {dataset_name.upper()}")
    print("="*80)
    
    # Select configuration
    if dataset_name == 'cifar10':
        config = CIFAR10Config()
        data_loader = create_cifar10_loaders(
            data_dir=str(config.RAW_DATA_DIR),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            validation_split=0.1,
            use_augmentation=config.USE_AUGMENTATION,
            input_size=config.INPUT_SIZE[0]
        )
    elif dataset_name == 'caltech101':
        config = Caltech101Config()
        data_loader = create_caltech101_loaders(
            data_dir=str(config.RAW_DATA_DIR),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            validation_split=0.2,
            test_split=0.2,
            use_augmentation=config.USE_AUGMENTATION,
            input_size=config.INPUT_SIZE[0]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Classes: {config.NUM_CLASSES}")
    print(f"  Input size: {config.INPUT_SIZE}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = create_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.5,
        pretrained=False
    )
    
    print_model_summary(model)
    
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
    
    return history


def main():
    """
    Train models on both datasets
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "COMPREHENSIVE CNN EVALUATION" + " "*30 + "║")
    print("║" + " "*15 + "Training on Multiple Datasets" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    # Configuration
    datasets = ['cifar10', 'caltech101']
    model_name = 'baseline'  # Change to 'improved', 'vgg', or 'resnet18' for better performance
    
    results_summary = {}
    
    for dataset in datasets:
        print(f"\n\n{'='*80}")
        print(f"DATASET {datasets.index(dataset) + 1}/{len(datasets)}: {dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            history = train_on_dataset(dataset, model_name)
            
            results_summary[dataset] = {
                'best_val_acc': max(history['val_acc']),
                'final_train_acc': history['train_acc'][-1],
                'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1
            }
            
            print(f"\n✓ {dataset.upper()} Training Complete!")
            print(f"  Best Val Acc: {results_summary[dataset]['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"\n✗ Error training on {dataset}: {e}")
            results_summary[dataset] = {'error': str(e)}
    
    # Print final summary
    print("\n\n" + "="*80)
    print(" "*25 + "FINAL RESULTS SUMMARY")
    print("="*80)
    
    for dataset, results in results_summary.items():
        print(f"\n{dataset.upper()}:")
        if 'error' in results:
            print(f"  ✗ Training failed: {results['error']}")
        else:
            print(f"  ✓ Best Validation Accuracy: {results['best_val_acc']:.2f}%")
            print(f"  ✓ Final Training Accuracy: {results['final_train_acc']:.2f}%")
            print(f"  ✓ Best Epoch: {results['best_epoch']}")
    
    # Save summary
    summary_path = Path('results/cnn/multi_dataset_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n\nResults summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    
    print("\nYou now have results for:")
    print("  ✓ CIFAR-10 (10 classes, 32x32 images)")
    print("  ✓ Caltech-101 (101 classes, 224x224 images)")
    print("\nUse these for Parts 1-2 of your coursework report!")


if __name__ == "__main__":
    main()
