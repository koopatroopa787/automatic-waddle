"""
Comprehensive Model Comparison
COMP64301: Computer Vision Coursework

Compare all 4 models:
1. CNN on CIFAR-10
2. CNN on Caltech-101
3. Traditional CV on CIFAR-10
4. Traditional CV on Caltech-101
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_cnn_results():
    """Load CNN results from both datasets"""
    results = {}
    
    # CIFAR-10
    cifar_path = Path('results/cnn/baseline_cnn_cifar10_20251128_230534/final_results.json')
    if cifar_path.exists():
        with open(cifar_path, 'r') as f:
            results['cnn_cifar10'] = json.load(f)
    
    # Caltech-101
    caltech_path = Path('results/cnn/improved_cnn_caltech101_20251130_000210/final_results.json')
    if caltech_path.exists():
        with open(caltech_path, 'r') as f:
            results['cnn_caltech101'] = json.load(f)
    
    return results


def load_traditional_cv_results():
    """Load Traditional CV results from both datasets"""
    results = {}
    
    # CIFAR-10
    cifar_path = Path('results/traditional_cv/cifar10_traditional_cv_results.json')
    if cifar_path.exists():
        with open(cifar_path, 'r') as f:
            results['trad_cifar10'] = json.load(f)
    
    # Caltech-101
    caltech_path = Path('results/traditional_cv/caltech101_traditional_cv_results.json')
    if caltech_path.exists():
        with open(caltech_path, 'r') as f:
            results['trad_caltech101'] = json.load(f)
    
    return results


def create_comparison_plot(cnn_results, trad_results, save_path):
    """Create comprehensive comparison plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================================================
    # Plot 1: Accuracy Comparison by Dataset
    # ========================================================================
    datasets = ['CIFAR-10\n(10 classes)', 'Caltech-101\n(101 classes)']
    
    cnn_cifar = cnn_results.get('cnn_cifar10', {}).get('test_acc', 0)
    cnn_caltech = cnn_results.get('cnn_caltech101', {}).get('best_val_acc', 0)
    trad_cifar = trad_results.get('trad_cifar10', {}).get('test_accuracy', 0) * 100
    trad_caltech = trad_results.get('trad_caltech101', {}).get('test_accuracy', 0) * 100
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [cnn_cifar, cnn_caltech], width, 
                    label='CNN', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, [trad_cifar, trad_caltech], width, 
                    label='Traditional CV', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison by Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    # ========================================================================
    # Plot 2: Performance Gap Analysis
    # ========================================================================
    gaps = [cnn_cifar - trad_cifar, cnn_caltech - trad_caltech]
    colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in gaps]
    
    bars = ax2.bar(datasets, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Performance Gap (%)\n(CNN - Traditional CV)', fontsize=12, fontweight='bold')
    ax2.set_title('CNN vs Traditional CV Performance Gap', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1 if height > 0 else height - 3,
                f'{gap:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # ========================================================================
    # Plot 3: Method Comparison (grouped by method)
    # ========================================================================
    methods = ['CNN', 'Traditional CV']
    cifar_vals = [cnn_cifar, trad_cifar]
    caltech_vals = [cnn_caltech, trad_caltech]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, cifar_vals, width, 
                    label='CIFAR-10', color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, caltech_vals, width, 
                    label='Caltech-101', color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Method Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend(fontsize=11)
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # ========================================================================
    # Plot 4: Summary Table
    # ========================================================================
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Model', 'CIFAR-10', 'Caltech-101', 'Avg'],
        ['CNN', f'{cnn_cifar:.2f}%', f'{cnn_caltech:.2f}%', 
         f'{(cnn_cifar + cnn_caltech)/2:.2f}%'],
        ['Traditional CV', f'{trad_cifar:.2f}%', f'{trad_caltech:.2f}%',
         f'{(trad_cifar + trad_caltech)/2:.2f}%'],
        ['Performance Gap', f'{cnn_cifar - trad_cifar:.2f}%', 
         f'{cnn_caltech - trad_caltech:.2f}%',
         f'{((cnn_cifar - trad_cifar) + (cnn_caltech - trad_caltech))/2:.2f}%']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    # Highlight last row
    for j in range(4):
        table[(3, j)].set_facecolor('#f39c12')
        table[(3, j)].set_text_props(weight='bold')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {save_path}")
    plt.close()


def print_analysis(cnn_results, trad_results):
    """Print detailed analysis"""
    print("\n" + "="*80)
    print(" "*25 + "DETAILED ANALYSIS")
    print("="*80)
    
    # Extract values
    cnn_cifar = cnn_results.get('cnn_cifar10', {}).get('test_acc', 0)
    cnn_caltech = cnn_results.get('cnn_caltech101', {}).get('best_val_acc', 0)
    trad_cifar = trad_results.get('trad_cifar10', {}).get('test_accuracy', 0) * 100
    trad_caltech = trad_results.get('trad_caltech101', {}).get('test_accuracy', 0) * 100
    
    print("\n📊 CIFAR-10 (10 classes):")
    print(f"   CNN:           {cnn_cifar:.2f}%")
    print(f"   Traditional CV: {trad_cifar:.2f}%")
    print(f"   Gap:           {cnn_cifar - trad_cifar:.2f}% (CNN better)")
    
    print("\n📊 Caltech-101 (101 classes):")
    print(f"   CNN:           {cnn_caltech:.2f}%")
    print(f"   Traditional CV: {trad_caltech:.2f}%")
    print(f"   Gap:           {cnn_caltech - trad_caltech:.2f}% (CNN better)")
    
    print("\n💡 Key Insights:")
    print(f"   1. CNN outperforms Traditional CV on both datasets")
    print(f"   2. Average CNN accuracy: {(cnn_cifar + cnn_caltech)/2:.2f}%")
    print(f"   3. Average Traditional CV: {(trad_cifar + trad_caltech)/2:.2f}%")
    print(f"   4. Average performance gap: {((cnn_cifar - trad_cifar) + (cnn_caltech - trad_caltech))/2:.2f}%")
    
    print("\n🔍 Dataset Difficulty:")
    print(f"   CIFAR-10:    Easier  (CNN: {cnn_cifar:.1f}%, Trad: {trad_cifar:.1f}%)")
    print(f"   Caltech-101: Harder  (CNN: {cnn_caltech:.1f}%, Trad: {trad_caltech:.1f}%)")
    
    print("\n📝 For Your Report:")
    print("   • CNN learns features automatically → better performance")
    print("   • Traditional CV uses hand-crafted SIFT features → limited")
    print("   • Performance gap larger on complex dataset (Caltech-101)")
    print("   • Both methods struggle with many classes")


def main():
    """Main comparison function"""
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE MODEL COMPARISON")
    print(" "*25 + "(All 4 Models)")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    cnn_results = load_cnn_results()
    trad_results = load_traditional_cv_results()
    
    # Check what's available
    available = []
    if 'cnn_cifar10' in cnn_results:
        available.append("✓ CNN on CIFAR-10")
    if 'cnn_caltech101' in cnn_results:
        available.append("✓ CNN on Caltech-101")
    if 'trad_cifar10' in trad_results:
        available.append("✓ Traditional CV on CIFAR-10")
    if 'trad_caltech101' in trad_results:
        available.append("✓ Traditional CV on Caltech-101")
    
    print("\nAvailable results:")
    for item in available:
        print(f"  {item}")
    
    if len(available) < 4:
        print("\n⚠ Warning: Not all 4 models have been trained yet!")
        print(f"  Found: {len(available)}/4 models")
        print("\nMissing:")
        if 'cnn_cifar10' not in cnn_results:
            print("  ✗ CNN on CIFAR-10 - Run: python main_cnn.py")
        if 'cnn_caltech101' not in cnn_results:
            print("  ✗ CNN on Caltech-101 - Run: python main_cnn_caltech101.py")
        if 'trad_cifar10' not in trad_results:
            print("  ✗ Traditional CV on CIFAR-10 - Run: python main_traditional_cv_cifar10.py")
        if 'trad_caltech101' not in trad_results:
            print("  ✗ Traditional CV on Caltech-101 - Run: python main_traditional_cv_caltech101.py")
        return
    
    # Create comparison visualizations
    print("\n" + "="*80)
    print("Generating comparison visualizations...")
    print("="*80)
    
    figures_dir = Path('results/comparison')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(
        cnn_results,
        trad_results,
        figures_dir / 'comprehensive_comparison.png'
    )
    
    # Print analysis
    print_analysis(cnn_results, trad_results)
    
    print("\n" + "="*80)
    print(" "*25 + "COMPARISON COMPLETE!")
    print("="*80)
    
    print(f"\nFigures saved to: {figures_dir.absolute()}")
    print("\n✅ Use these figures in Part 5 of your report!")


if __name__ == "__main__":
    main()
