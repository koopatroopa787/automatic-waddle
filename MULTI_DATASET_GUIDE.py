"""
Multi-Dataset Training Guide
COMP64301: Computer Vision Coursework

Complete guide for training on CIFAR-10 and Caltech-101
"""

MULTI_DATASET_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              MULTI-DATASET CNN TRAINING GUIDE (2 DATASETS)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Your project now supports TWO datasets for comprehensive evaluation!

════════════════════════════════════════════════════════════════════════════════
📊 DATASET STRATEGY
════════════════════════════════════════════════════════════════════════════════

DATASET 1: CIFAR-10 (Primary)
─────────────────────────────────
• Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
• Images: 60,000 (50k train, 10k test)
• Resolution: 32×32 RGB
• Size: ~170 MB download
• Characteristics:
  ✓ Low resolution - tests model on small images
  ✓ Balanced classes - 6000 images per class
  ✓ Fast to train - good for experimentation
  ✓ Standard benchmark - easy to compare with literature

DATASET 2: Caltech-101 (Secondary)
─────────────────────────────────
• Classes: 101 (various object categories)
• Images: ~9,000 total
• Resolution: Variable (~300×200 average, resized to 224×224)
• Size: ~130 MB download
• Characteristics:
  ✓ High resolution - tests model on realistic images
  ✓ More classes - tests model capacity
  ✓ Imbalanced - 40-800 images per class (realistic scenario)
  ✓ Variable sizes - tests preprocessing robustness

════════════════════════════════════════════════════════════════════════════════
🎯 WHY THIS COMBINATION?
════════════════════════════════════════════════════════════════════════════════

1. DIFFERENT SCALES:
   • CIFAR-10: Small images (32×32)
   • Caltech-101: Large images (224×224)
   → Shows your models work across different resolutions

2. DIFFERENT COMPLEXITY:
   • CIFAR-10: 10 classes (simpler)
   • Caltech-101: 101 classes (more complex)
   → Tests model capacity and generalization

3. DIFFERENT DISTRIBUTIONS:
   • CIFAR-10: Balanced, uniform distribution
   • Caltech-101: Imbalanced, realistic distribution
   → Shows robustness to data characteristics

4. FAIR COMPARISON:
   • Both CNN and Traditional CV will use BOTH datasets
   • This gives you 4 total models to compare:
     - CNN on CIFAR-10
     - CNN on Caltech-101
     - Traditional CV on CIFAR-10
     - Traditional CV on Caltech-101

════════════════════════════════════════════════════════════════════════════════
🚀 QUICK START - TRAIN ON BOTH DATASETS
════════════════════════════════════════════════════════════════════════════════

OPTION 1: Train Both Datasets in One Go (Recommended)
────────────────────────────────────────────────────

    python train_all_datasets.py

This will:
✓ Train on CIFAR-10 first
✓ Then train on Caltech-101
✓ Save results for both
✓ Generate comparison summary


OPTION 2: Train Each Dataset Separately
────────────────────────────────────────

For CIFAR-10:
    python main_cnn.py

For Caltech-101:
    python main_cnn_caltech101.py


════════════════════════════════════════════════════════════════════════════════
⏱️  TRAINING TIME ESTIMATES
════════════════════════════════════════════════════════════════════════════════

CIFAR-10 (50 epochs):
  • CPU: 20-30 minutes
  • GPU: 5-10 minutes

Caltech-101 (50 epochs):
  • CPU: 30-50 minutes (larger images)
  • GPU: 10-15 minutes

TOTAL for both: 1-2 hours on CPU, 15-25 minutes on GPU

════════════════════════════════════════════════════════════════════════════════
📈 EXPECTED PERFORMANCE
════════════════════════════════════════════════════════════════════════════════

CIFAR-10:
┌─────────────┬──────────────┬────────────┐
│ Model       │ Test Acc     │ Parameters │
├─────────────┼──────────────┼────────────┤
│ BaselineCNN │ 75-80%       │ ~1.2M      │
│ ImprovedCNN │ 80-85%       │ ~1.8M      │
│ ResNet18    │ 85-90%       │ ~11M       │
└─────────────┴──────────────┴────────────┘

Caltech-101:
┌─────────────┬──────────────┬────────────┐
│ Model       │ Test Acc     │ Parameters │
├─────────────┼──────────────┼────────────┤
│ BaselineCNN │ 50-60%       │ ~1.2M      │
│ ImprovedCNN │ 60-70%       │ ~1.8M      │
│ ResNet18    │ 70-80%       │ ~11M       │
└─────────────┴──────────────┴────────────┘

Note: Caltech-101 is harder (more classes, less data per class)

════════════════════════════════════════════════════════════════════════════════
🔧 CUSTOMIZATION FOR EACH DATASET
════════════════════════════════════════════════════════════════════════════════

configs/config.py contains separate configurations:

CIFAR10Config:
  • INPUT_SIZE = (32, 32)
  • NUM_CLASSES = 10
  • BATCH_SIZE = 32
  • Use native resolution

Caltech101Config:
  • INPUT_SIZE = (224, 224)
  • NUM_CLASSES = 101
  • BATCH_SIZE = 32
  • Higher resolution for better feature extraction

════════════════════════════════════════════════════════════════════════════════
📝 FOR YOUR COURSEWORK REPORT
════════════════════════════════════════════════════════════════════════════════

Having 2 datasets allows you to discuss:

1. GENERALIZATION:
   • How well do models transfer across datasets?
   • Which architecture is most robust?

2. SCALABILITY:
   • How does performance change with more classes?
   • Does the model handle variable resolution?

3. DATA EFFICIENCY:
   • CIFAR-10 has more samples per class
   • Caltech-101 has fewer samples per class
   • How does this affect learning?

4. COMPARISON DEPTH:
   • Compare CNN vs Traditional CV on BOTH datasets
   • Which approach works better on each?
   • Why might one be better for certain characteristics?

════════════════════════════════════════════════════════════════════════════════
🎓 COURSEWORK STRUCTURE WITH 2 DATASETS
════════════════════════════════════════════════════════════════════════════════

Part 1-2: CNN Approach (16 marks)
  ├── CIFAR-10 experiments
  │   ├── Baseline model
  │   ├── Hyperparameter tuning
  │   └── Results analysis
  │
  └── Caltech-101 experiments
      ├── Same architecture
      ├── Adapted hyperparameters
      └── Comparative analysis

Part 3-4: Traditional CV Approach (16 marks)
  ├── CIFAR-10 experiments
  │   ├── SIFT/ORB features
  │   ├── Bag-of-Words
  │   └── SVM/KNN classification
  │
  └── Caltech-101 experiments
      ├── Same pipeline
      ├── Parameter tuning
      └── Comparative analysis

Part 5: Comparison (3 marks)
  └── CNN vs Traditional CV on BOTH datasets
      ├── Which works better where?
      ├── Why?
      └── Trade-offs

════════════════════════════════════════════════════════════════════════════════
💡 PRO TIPS
════════════════════════════════════════════════════════════════════════════════

1. START WITH CIFAR-10:
   ✓ Faster to train
   ✓ Good for debugging
   ✓ Test hyperparameters here first

2. USE INSIGHTS FROM CIFAR-10 FOR CALTECH-101:
   ✓ Best hyperparameters from CIFAR-10
   ✓ Apply to Caltech-101 with minor adjustments
   ✓ Document what transfers and what doesn't

3. MODEL SELECTION:
   ✓ Baseline: Good for understanding fundamentals
   ✓ Improved: Better balance of performance/complexity
   ✓ ResNet18: Best performance, use for final results

4. FOR CALTECH-101:
   ✓ Use higher learning rate (more classes to learn)
   ✓ More dropout (prevent overfitting on small dataset)
   ✓ More augmentation (limited data per class)

════════════════════════════════════════════════════════════════════════════════
📊 RESULTS ORGANIZATION
════════════════════════════════════════════════════════════════════════════════

results/
├── cnn/
│   ├── baseline_cifar10_*/          # CIFAR-10 experiments
│   │   ├── best_model.pth
│   │   ├── training_history.json
│   │   └── final_results.json
│   │
│   ├── improved_caltech101_*/       # Caltech-101 experiments
│   │   ├── best_model.pth
│   │   ├── training_history.json
│   │   └── final_results.json
│   │
│   └── multi_dataset_summary.json   # Overall comparison
│
└── comparison/                       # Cross-dataset analysis
    └── (will be created later)

════════════════════════════════════════════════════════════════════════════════
✅ COMPLETE WORKFLOW
════════════════════════════════════════════════════════════════════════════════

WEEK 1: CNN Implementation
  Day 1-2: Train on CIFAR-10
    □ python main_cnn.py
    □ python hyperparameter_tuning.py
  
  Day 3-4: Train on Caltech-101
    □ python main_cnn_caltech101.py
    □ Adapt hyperparameters
  
  Day 5: Compare results
    □ Generate visualizations
    □ Create comparison tables

WEEK 2: Traditional CV Implementation
  Day 1-2: Implement on CIFAR-10
    □ SIFT/ORB extraction
    □ Bag-of-Words
    □ Classification
  
  Day 3-4: Implement on Caltech-101
    □ Same pipeline
    □ Parameter tuning
  
  Day 5: Compare CNN vs Traditional CV

WEEK 3: Report Writing
  □ Write methods sections
  □ Create figures and tables
  □ Analysis and discussion
  □ State-of-the-art review

════════════════════════════════════════════════════════════════════════════════
🎯 SUCCESS CRITERIA
════════════════════════════════════════════════════════════════════════════════

✓ Train CNN on BOTH datasets
✓ Implement Traditional CV on BOTH datasets
✓ Fair comparison with same random seeds
✓ Document differences in approach for each dataset
✓ Analyze why certain methods work better
✓ Generate high-quality visualizations
✓ Write clear, justified report

════════════════════════════════════════════════════════════════════════════════

Good luck! You now have a comprehensive framework for multi-dataset evaluation! 🚀

Questions? Check:
  • README.md for project overview
  • CNN_USAGE_GUIDE.py for detailed CNN instructions
  • Discussion forum on Canvas
"""

if __name__ == "__main__":
    print(MULTI_DATASET_GUIDE)
