"""
Comprehensive Results Visualization
COMP64301: Computer Vision Coursework

Visualizes results from both CIFAR-10 and Caltech-101 experiments
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(experiment_dir):
    """Load training history and final results"""
    exp_path = Path(experiment_dir)
    
    # Load training history
    history_file = exp_path / 'training_history.json'
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Load final results
    results_file = exp_path / 'final_results.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return history, results


def plot_training_curves(history, title, save_path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title} - Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comparison(cifar_history, caltech_history, save_path):
    """Plot comparison between CIFAR-10 and Caltech-101"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_cifar = range(1, len(cifar_history['val_acc']) + 1)
    epochs_caltech = range(1, len(caltech_history['val_acc']) + 1)
    
    # Validation accuracy comparison
    ax1.plot(epochs_cifar, cifar_history['val_acc'], 'b-', 
             label='CIFAR-10 (10 classes)', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs_caltech, caltech_history['val_acc'], 'r-', 
             label='Caltech-101 (101 classes)', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Final performance bar chart
    datasets = ['CIFAR-10\n(10 classes)', 'Caltech-101\n(101 classes)']
    accuracies = [max(cifar_history['val_acc']), max(caltech_history['val_acc'])]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax2.bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_summary_table(cifar_results, caltech_results, save_path):
    """Create summary table comparing results"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'CIFAR-10', 'Caltech-101'],
        ['Dataset Size', '60,000 images', '~9,000 images'],
        ['Number of Classes', '10', '101'],
        ['Input Size', '32×32', '224×224'],
        ['Model', 'Baseline CNN', 'Improved CNN'],
        ['Parameters', '~650K', '~1.2M'],
        ['Best Val Accuracy', f"{cifar_results['best_val_acc']:.2f}%", 
         f"{caltech_results['best_val_acc']:.2f}%"],
        ['Test Accuracy', f"{cifar_results['test_acc']:.2f}%", 'N/A'],
        ['Training Time', f"{cifar_results['training_time_minutes']:.1f} min", 
         f"{caltech_results['training_time_minutes']:.1f} min"],
        ['Best Epoch', str(cifar_results['best_epoch']), str(caltech_results['best_epoch'])]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('CNN Results Summary - CIFAR-10 vs Caltech-101', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_learning_rate_schedule(history, title, save_path):
    """Plot learning rate schedule"""
    if 'learning_rates' not in history or not history['learning_rates']:
        print(f"  ⚠ No learning rate data for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history['learning_rates']) + 1)
    ax.plot(epochs, history['learning_rates'], 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main visualization function"""
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE RESULTS VISUALIZATION")
    print("="*80 + "\n")
    
    # Define experiment directories
    cifar_exp = 'results/cnn/baseline_cnn_cifar10_20251128_230534'
    caltech_exp = 'results/cnn/improved_cnn_caltech101_20251130_000210'
    
    # Create figures directory
    figures_dir = Path('results/cnn/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    print("-" * 80)
    
    # Load results
    try:
        cifar_history, cifar_results = load_results(cifar_exp)
        print(f"✓ Loaded CIFAR-10 results: {cifar_results['best_val_acc']:.2f}% val acc")
    except Exception as e:
        print(f"✗ Error loading CIFAR-10 results: {e}")
        return
    
    try:
        caltech_history, caltech_results = load_results(caltech_exp)
        print(f"✓ Loaded Caltech-101 results: {caltech_results['best_val_acc']:.2f}% val acc")
    except Exception as e:
        print(f"✗ Error loading Caltech-101 results: {e}")
        return
    
    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80 + "\n")
    
    # 1. Individual training curves
    print("1. Creating training curves...")
    plot_training_curves(
        cifar_history, 
        'CIFAR-10 (Baseline CNN)',
        figures_dir / 'cifar10_training_curves.png'
    )
    plot_training_curves(
        caltech_history, 
        'Caltech-101 (Improved CNN)',
        figures_dir / 'caltech101_training_curves.png'
    )
    
    # 2. Comparison plot
    print("\n2. Creating comparison plot...")
    plot_comparison(
        cifar_history,
        caltech_history,
        figures_dir / 'dataset_comparison.png'
    )
    
    # 3. Summary table
    print("\n3. Creating summary table...")
    create_summary_table(
        cifar_results,
        caltech_results,
        figures_dir / 'results_summary_table.png'
    )
    
    # 4. Learning rate schedules
    print("\n4. Creating learning rate schedules...")
    plot_learning_rate_schedule(
        cifar_history,
        'CIFAR-10',
        figures_dir / 'cifar10_lr_schedule.png'
    )
    plot_learning_rate_schedule(
        caltech_history,
        'Caltech-101',
        figures_dir / 'caltech101_lr_schedule.png'
    )
    
    # Print summary
    print("\n" + "="*80)
    print(" "*25 + "VISUALIZATION COMPLETE!")
    print("="*80)
    
    print("\nGenerated Files:")
    print("  📊 cifar10_training_curves.png - CIFAR-10 loss and accuracy curves")
    print("  📊 caltech101_training_curves.png - Caltech-101 loss and accuracy curves")
    print("  📊 dataset_comparison.png - Side-by-side comparison")
    print("  📊 results_summary_table.png - Summary table")
    print("  📊 cifar10_lr_schedule.png - Learning rate schedule")
    print("  📊 caltech101_lr_schedule.png - Learning rate schedule")
    
    print(f"\nAll figures saved to: {figures_dir.absolute()}")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 CIFAR-10 (Baseline CNN):")
    print(f"   Best Validation Accuracy: {cifar_results['best_val_acc']:.2f}%")
    print(f"   Test Accuracy: {cifar_results['test_acc']:.2f}%")
    print(f"   Training Time: {cifar_results['training_time_minutes']:.1f} minutes")
    print(f"   Best Epoch: {cifar_results['best_epoch']}")
    
    print("\n📊 Caltech-101 (Improved CNN):")
    print(f"   Best Validation Accuracy: {caltech_results['best_val_acc']:.2f}%")
    print(f"   Training Time: {caltech_results['training_time_minutes']:.1f} minutes")
    print(f"   Best Epoch: {caltech_results['best_epoch']}")
    
    print("\n💡 Analysis:")
    print("   • CIFAR-10 achieved 79.67% - Strong baseline performance!")
    print("   • Caltech-101 achieved 46.42% - Reasonable for 101 classes with limited data")
    print("   • CIFAR-10 is easier: fewer classes, more balanced data")
    print("   • Caltech-101 is harder: 10x more classes, fewer samples per class")
    
    print("\n✅ Next Steps:")
    print("   1. Use these figures in your report (Parts 1-2)")
    print("   2. Implement Traditional CV methods (Parts 3-4)")
    print("   3. Compare CNN vs Traditional CV (Part 5)")
    print("   4. Write state-of-the-art review (Part 6)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
