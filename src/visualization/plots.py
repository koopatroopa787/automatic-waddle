"""
Visualization Utilities
COMP64301: Computer Vision Coursework

Functions for visualizing training results, metrics, and model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_learning_rate(history, save_path=None):
    """
    Plot learning rate schedule
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    if 'learning_rates' not in history:
        print("No learning rate information in history")
        return
    
    epochs = range(1, len(history['learning_rates']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to: {save_path}")
    
    plt.show()


def plot_hyperparameter_comparison(results_dict, metric='val_acc', save_path=None):
    """
    Compare different hyperparameter configurations
    
    Args:
        results_dict: Dictionary with config names as keys and histories as values
        metric: Metric to compare ('val_acc' or 'val_loss')
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for name, history in results_dict.items():
        epochs = range(1, len(history[metric]) + 1)
        plt.plot(epochs, history[metric], label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    
    if metric == 'val_acc':
        plt.ylabel('Validation Accuracy (%)', fontsize=12)
        plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    else:
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def create_results_summary_table(results_dir, save_path=None):
    """
    Create a summary table of all experiments
    
    Args:
        results_dir: Directory containing experiment results
        save_path: Optional path to save the table
    """
    results_dir = Path(results_dir)
    
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / 'final_results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                experiments.append({
                    'Experiment': exp_dir.name,
                    'Best Val Acc': f"{results['best_val_acc']:.2f}%",
                    'Test Acc': f"{results['test_acc']:.2f}%",
                    'Best Epoch': results['best_epoch'],
                    'Training Time': f"{results['training_time_minutes']:.1f} min"
                })
    
    if not experiments:
        print("No experiment results found")
        return
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, len(experiments) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=[[exp[k] for k in experiments[0].keys()] for exp in experiments],
        colLabels=list(experiments[0].keys()),
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(experiments[0].keys())):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results summary saved to: {save_path}")
    
    plt.show()


def visualize_experiment_results(experiment_dir):
    """
    Load and visualize results from an experiment
    
    Args:
        experiment_dir: Path to experiment directory
    """
    experiment_dir = Path(experiment_dir)
    
    # Load training history
    history_file = experiment_dir / 'training_history.json'
    
    if not history_file.exists():
        print(f"No training history found in {experiment_dir}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figures directory
    figures_dir = experiment_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(
        history,
        save_path=figures_dir / 'training_curves.png'
    )
    
    # Plot learning rate
    print("Plotting learning rate...")
    plot_learning_rate(
        history,
        save_path=figures_dir / 'learning_rate.png'
    )
    
    print(f"\nVisualizations saved to: {figures_dir}")


if __name__ == "__main__":
    print("Visualization Module")
    print("Import this module to create plots and visualizations")
