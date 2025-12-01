"""
Hyperparameter Tuning Script
COMP64301: Computer Vision Coursework

Systematic exploration of hyperparameters for CNN models.
"""

import sys
from pathlib import Path
import copy

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from configs.config import CIFAR10Config
from src.cnn.data_loader import create_cifar10_loaders
from src.cnn.models import create_model
from src.cnn.train import train_model
from src.utils.helpers import set_seed


class HyperparameterTuner:
    """
    Systematic hyperparameter tuning for CNN models
    """
    
    def __init__(self, base_config, param_grid):
        """
        Initialize tuner
        
        Args:
            base_config: Base configuration object
            param_grid: Dictionary of parameters to tune
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.results = []
    
    def tune(self, data_loaders, model_name='baseline'):
        """
        Run hyperparameter tuning experiments
        
        Args:
            data_loaders: Tuple of (train_loader, val_loader, test_loader)
            model_name: Name of model architecture to use
        """
        train_loader, val_loader, test_loader = data_loaders
        
        print("\n" + "="*80)
        print(" "*20 + "HYPERPARAMETER TUNING")
        print("="*80)
        
        print("\nParameter Grid:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        
        total_experiments = self._count_experiments()
        print(f"\nTotal experiments to run: {total_experiments}")
        
        experiment_num = 0
        
        # Iterate through all combinations
        for param_name, param_values in self.param_grid.items():
            for value in param_values:
                experiment_num += 1
                
                print("\n" + "-"*80)
                print(f"Experiment {experiment_num}/{total_experiments}")
                print(f"Testing: {param_name} = {value}")
                print("-"*80)
                
                # Create modified config
                config = copy.deepcopy(self.base_config)
                setattr(config, param_name, value)
                
                # Set random seed for fair comparison
                set_seed(config.RANDOM_SEED)
                
                # Create model
                if param_name == 'DROPOUT_RATE':
                    model = create_model(
                        model_name=model_name,
                        num_classes=config.NUM_CLASSES,
                        dropout_rate=value
                    )
                else:
                    model = create_model(
                        model_name=model_name,
                        num_classes=config.NUM_CLASSES
                    )
                
                # Train model
                experiment_name = f'{param_name}_{value}'
                trained_model, history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config,
                    experiment_name=experiment_name
                )
                
                # Store results
                self.results.append({
                    'param_name': param_name,
                    'param_value': value,
                    'best_val_acc': max(history['val_acc']),
                    'final_train_acc': history['train_acc'][-1],
                    'experiment_name': experiment_name
                })
        
        self._print_summary()
        
        return self.results
    
    def _count_experiments(self):
        """Count total number of experiments"""
        return sum(len(values) for values in self.param_grid.values())
    
    def _print_summary(self):
        """Print summary of tuning results"""
        print("\n" + "="*80)
        print(" "*20 + "HYPERPARAMETER TUNING SUMMARY")
        print("="*80)
        
        for param_name in self.param_grid.keys():
            print(f"\n{param_name}:")
            print("-" * 40)
            
            param_results = [r for r in self.results if r['param_name'] == param_name]
            
            for result in sorted(param_results, key=lambda x: x['best_val_acc'], reverse=True):
                print(f"  {result['param_value']}: {result['best_val_acc']:.2f}% val acc")
        
        # Find best overall
        best_result = max(self.results, key=lambda x: x['best_val_acc'])
        print("\n" + "="*80)
        print("BEST CONFIGURATION:")
        print(f"  {best_result['param_name']} = {best_result['param_value']}")
        print(f"  Validation Accuracy: {best_result['best_val_acc']:.2f}%")
        print("="*80)


def main():
    """
    Main function for hyperparameter tuning
    """
    # Base configuration
    config = CIFAR10Config()
    config.EPOCHS = 30  # Reduce epochs for faster tuning
    
    # Load data once
    print("\nLoading CIFAR-10 Dataset...")
    data_loader = create_cifar10_loaders(
        data_dir=str(config.RAW_DATA_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=0.1,
        use_augmentation=config.USE_AUGMENTATION
    )
    
    loaders = data_loader.get_loaders()
    
    # Define parameter grid for tuning
    # Focus on key parameters that impact performance
    param_grid = {
        'LEARNING_RATE': [0.001, 0.01, 0.1],
        # 'BATCH_SIZE': [32, 64, 128],
        # 'WEIGHT_DECAY': [1e-4, 1e-3, 1e-2],
        # 'DROPOUT_RATE': [0.3, 0.5, 0.7],
    }
    
    # Create tuner
    tuner = HyperparameterTuner(config, param_grid)
    
    # Run tuning
    results = tuner.tune(loaders, model_name='baseline')
    
    print("\nHyperparameter tuning complete!")
    print("Results saved in: results/cnn/")


if __name__ == "__main__":
    main()
