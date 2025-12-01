"""
CNN Implementation Usage Guide
COMP64301: Computer Vision Coursework

This guide explains how to use the CNN implementation for your coursework.
"""

USAGE_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CNN IMPLEMENTATION USAGE GUIDE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

The CNN implementation is complete! Here's how to use it for your coursework.

════════════════════════════════════════════════════════════════════════════════
📦 STEP 1: INSTALL DEPENDENCIES
════════════════════════════════════════════════════════════════════════════════

Navigate to the project directory and install requirements:

    cd cv_coursework
    pip install -r requirements.txt

This will install:
    • PyTorch and torchvision (deep learning)
    • OpenCV (computer vision)
    • scikit-learn (traditional ML)
    • matplotlib, seaborn (visualization)
    • And other necessary packages

════════════════════════════════════════════════════════════════════════════════
🚀 STEP 2: TRAIN BASELINE CNN
════════════════════════════════════════════════════════════════════════════════

Run the main training script:

    python main_cnn.py

This will:
    ✓ Download CIFAR-10 dataset (~170 MB)
    ✓ Create train/validation/test splits
    ✓ Train a baseline CNN model
    ✓ Save the best model
    ✓ Generate training metrics
    ✓ Test the final model

Expected output:
    • Best model saved to: models/cnn/best_model.pth
    • Training history: results/cnn/training_history.json
    • Final results: results/cnn/final_results.json

Training time: ~15-30 minutes on CPU, ~5-10 minutes on GPU

════════════════════════════════════════════════════════════════════════════════
🔧 STEP 3: HYPERPARAMETER TUNING
════════════════════════════════════════════════════════════════════════════════

Run systematic hyperparameter exploration:

    python hyperparameter_tuning.py

This script will test different configurations:
    • Learning rates: [0.001, 0.01, 0.1]
    • Batch sizes: [32, 64, 128]
    • Weight decay: [1e-4, 1e-3, 1e-2]
    • Dropout rates: [0.3, 0.5, 0.7]

You can modify the param_grid in hyperparameter_tuning.py to test:
    • Different architectures ('baseline', 'improved', 'vgg', 'resnet18')
    • Optimizer parameters
    • Data augmentation settings
    • Input image sizes

════════════════════════════════════════════════════════════════════════════════
📊 STEP 4: VISUALIZE RESULTS
════════════════════════════════════════════════════════════════════════════════

Create visualizations for your report:

    from src.visualization.plots import (
        plot_training_curves,
        plot_hyperparameter_comparison,
        visualize_experiment_results
    )
    
    # Visualize a single experiment
    visualize_experiment_results('results/cnn/baseline_cnn_cifar10_20241128_120000')
    
    # Compare multiple experiments
    results = {
        'LR=0.001': history1,
        'LR=0.01': history2,
        'LR=0.1': history3
    }
    plot_hyperparameter_comparison(results, metric='val_acc')

This generates:
    • Training/validation curves
    • Learning rate schedules
    • Comparison plots
    • Confusion matrices

════════════════════════════════════════════════════════════════════════════════
🎨 CUSTOMIZATION OPTIONS
════════════════════════════════════════════════════════════════════════════════

1. CHANGE MODEL ARCHITECTURE:
   
   Edit main_cnn.py:
   
   model = create_model(
       model_name='improved',  # Try: 'baseline', 'improved', 'vgg', 'resnet18'
       num_classes=10,
       dropout_rate=0.5
   )

2. MODIFY DATA AUGMENTATION:
   
   Edit configs/config.py:
   
   AUGMENTATION_PARAMS = {
       'horizontal_flip': True,
       'vertical_flip': False,
       'rotation_range': 15,
       'zoom_range': 0.1,
   }

3. ADJUST TRAINING SETTINGS:
   
   Edit configs/config.py:
   
   EPOCHS = 50
   LEARNING_RATE = 0.001
   BATCH_SIZE = 64
   WEIGHT_DECAY = 1e-4

4. USE DIFFERENT INPUT SIZE:
   
   data_loader = create_cifar10_loaders(
       input_size=64  # Upscale from 32x32 to 64x64
   )

════════════════════════════════════════════════════════════════════════════════
📝 FOR YOUR REPORT (Parts 1-2)
════════════════════════════════════════════════════════════════════════════════

WHAT TO INCLUDE:

1. METHODS SECTION (Part 1: 8 marks):
   • Network architecture description
   • Justification for design choices
   • Hyperparameters tested and why
   • Data preprocessing steps
   • Training procedure
   
   Use: print_model_summary(model) to get architecture details

2. RESULTS SECTION (Part 2: 8 marks):
   • Training curves (loss and accuracy)
   • Hyperparameter exploration results
   • Best configuration found
   • Test set performance
   • Comparison tables
   
   Use: plot_training_curves() and plot_hyperparameter_comparison()

TIPS:
   ✓ Focus on UNDERSTANDING, not just results
   ✓ Explain WHY you chose certain hyperparameters
   ✓ Discuss what worked and what didn't
   ✓ Use appropriate figures and tables
   ✓ Reference figures in your text

════════════════════════════════════════════════════════════════════════════════
💡 QUICK EXPERIMENTS TO TRY
════════════════════════════════════════════════════════════════════════════════

1. BASELINE COMPARISON:
   • Train baseline, improved, vgg, and resnet18
   • Compare performance and parameters
   • Analyze trade-offs

2. LEARNING RATE STUDY:
   • Test: [0.0001, 0.001, 0.01, 0.1]
   • Plot validation accuracy vs learning rate
   • Find optimal value

3. DROPOUT EFFECT:
   • Test: [0.0, 0.3, 0.5, 0.7]
   • Compare train vs validation accuracy
   • Analyze overfitting

4. BATCH SIZE IMPACT:
   • Test: [16, 32, 64, 128]
   • Compare training time and accuracy
   • Discuss memory vs performance

5. DATA AUGMENTATION:
   • Train with and without augmentation
   • Compare generalization
   • Show example augmented images

════════════════════════════════════════════════════════════════════════════════
🐛 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

ISSUE: Out of memory error
FIX: Reduce batch size in configs/config.py

ISSUE: Training too slow
FIX: Use GPU if available, or reduce epochs/model size

ISSUE: Poor accuracy
FIX: Check data loading, try different learning rates, add augmentation

ISSUE: Model overfitting
FIX: Increase dropout, add weight decay, use data augmentation

ISSUE: Import errors
FIX: Make sure you're in the project root directory when running scripts

════════════════════════════════════════════════════════════════════════════════
📁 UNDERSTANDING THE OUTPUT
════════════════════════════════════════════════════════════════════════════════

After training, you'll find:

results/cnn/
├── baseline_cnn_cifar10_20241128_120000/
│   ├── best_model.pth              # Best model weights
│   ├── training_history.json       # Loss/accuracy per epoch
│   ├── final_results.json          # Summary statistics
│   └── figures/                    # Generated plots
│       ├── training_curves.png
│       └── learning_rate.png

models/cnn/
└── best_model.pth                  # Copy of best model

Use these files for:
    • Loading trained models
    • Creating report figures
    • Analyzing hyperparameter effects
    • Comparing different experiments

════════════════════════════════════════════════════════════════════════════════
✅ CHECKLIST FOR COURSEWORK PARTS 1-2
════════════════════════════════════════════════════════════════════════════════

Part 1: CNN Methods (8 marks)
□ Train baseline CNN on CIFAR-10
□ Document network architecture
□ Test at least 3 hyperparameters
□ Justify all design choices
□ Describe data preprocessing
□ Explain training procedure

Part 2: CNN Results (8 marks)
□ Create training/validation curves
□ Present hyperparameter results
□ Include comparison tables/charts
□ Interpret the results
□ Discuss what worked/didn't work
□ Suggest future improvements

════════════════════════════════════════════════════════════════════════════════
🎯 NEXT STEPS
════════════════════════════════════════════════════════════════════════════════

1. Run main_cnn.py to train baseline model
2. Run hyperparameter_tuning.py for systematic exploration
3. Create visualizations for your report
4. Document your findings
5. Move on to Traditional CV implementation (Parts 3-4)

════════════════════════════════════════════════════════════════════════════════

Good luck with your coursework! 🚀

For questions, refer to:
    • README.md for project overview
    • Discussion forum on Canvas
    • TAs: George Bird, Kai Cao
"""

if __name__ == "__main__":
    print(USAGE_GUIDE)
