"""
Utility Functions
COMP64301: Computer Vision Coursework
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get available computing device
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def save_model(model, filepath, metadata=None):
    """
    Save model with metadata
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        metadata: Optional dictionary with additional information
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if metadata:
        save_dict.update(metadata)
    
    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_model(model, filepath):
    """
    Load model from file
    
    Args:
        model: PyTorch model instance
        filepath: Path to saved model
        
    Returns:
        Loaded model and metadata
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    print(f"Model loaded from: {filepath}")
    return model, metadata


def save_results(results, filepath):
    """
    Save results dictionary to JSON file
    
    Args:
        results: Dictionary containing results
        filepath: Path to save results
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {filepath}")


def load_results(filepath):
    """
    Load results from JSON file
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary containing results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from: {filepath}")
    return results


def count_parameters(model):
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model, input_size=None):
    """
    Print summary of model architecture
    
    Args:
        model: PyTorch model
        input_size: Optional tuple of input dimensions (B, C, H, W)
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    if input_size:
        try:
            from torchsummary import summary
            summary(model, input_size)
        except ImportError:
            print("\nInstall torchsummary for detailed architecture summary:")
            print("pip install torchsummary")
    
    print("="*60 + "\n")


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking metrics during training
    """
    def __init__(self, name='metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a unique directory for experiment results
    
    Args:
        base_dir: Base directory path
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir
