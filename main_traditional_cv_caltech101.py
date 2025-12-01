"""
Traditional CV Training on Caltech-101
COMP64301: Computer Vision Coursework

Complete pipeline: Feature Extraction → Bag-of-Words → Classification
"""

import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.traditional_cv.feature_extraction import FeatureExtractor, create_caltech101_raw_dataset
from src.traditional_cv.bag_of_words import BagOfWords
from src.traditional_cv.classification import ImageClassifier
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    """Main training pipeline for Caltech-101"""
    print("\n" + "="*80)
    print(" "*15 + "TRADITIONAL CV ON CALTECH-101")
    print(" "*10 + "(SIFT + Bag-of-Words + SVM)")
    print("="*80)
    
    # Configuration
    FEATURE_TYPE = 'SIFT'  # or 'ORB'
    MAX_FEATURES = 200
    VOCAB_SIZE = 500
    CLASSIFIER_TYPE = 'LinearSVM'  # or 'SVM', 'RandomForest'
    TEST_SIZE = 0.2  # 20% for test
    
    # Paths
    results_dir = Path('results/traditional_cv')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading Caltech-101 Dataset")
    print("-"*80)
    
    dataset = create_caltech101_raw_dataset()
    
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Number of classes: 101")
    
    # ========================================================================
    # STEP 2: Extract Features
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Extracting Features")
    print("-"*80)
    print("Note: This may take longer than CIFAR-10 due to larger images")
    
    feature_extractor = FeatureExtractor(
        feature_type=FEATURE_TYPE,
        max_features=MAX_FEATURES
    )
    
    # Extract features from all images
    print("\nExtracting features from all images...")
    all_descriptors, all_labels = feature_extractor.extract_from_dataset(
        dataset,
        max_images=None  # Use all images
    )
    
    print(f"\n✓ Total images processed: {len(all_descriptors)}")
    print(f"✓ Total features extracted: {sum(len(d) for d in all_descriptors):,}")
    
    # Split into train and test
    print(f"\nSplitting into train ({100-int(TEST_SIZE*100)}%) and test ({int(TEST_SIZE*100)}%)...")
    
    indices = np.arange(len(all_descriptors))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=all_labels
    )
    
    train_descriptors = [all_descriptors[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    
    test_descriptors = [all_descriptors[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]
    
    print(f"✓ Training samples: {len(train_descriptors)}")
    print(f"✓ Test samples: {len(test_descriptors)}")
    
    # Save features
    feature_extractor.save_features(
        train_descriptors,
        train_labels,
        results_dir / f'caltech101_train_{FEATURE_TYPE}_features.pkl'
    )
    feature_extractor.save_features(
        test_descriptors,
        test_labels,
        results_dir / f'caltech101_test_{FEATURE_TYPE}_features.pkl'
    )
    
    # ========================================================================
    # STEP 3: Build Vocabulary (Bag-of-Words)
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Building Visual Vocabulary")
    print("-"*80)
    
    bow = BagOfWords(vocab_size=VOCAB_SIZE, random_state=42)
    
    # Use subset of descriptors for faster clustering
    # Caltech-101 has larger images, so more features per image
    bow.build_vocabulary(train_descriptors, max_samples=100000)
    
    # Save BoW model
    bow.save_model(results_dir / f'caltech101_bow_{VOCAB_SIZE}.pkl')
    
    # ========================================================================
    # STEP 4: Encode Images as Histograms
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Encoding Images")
    print("-"*80)
    
    # Encode training set
    X_train = bow.encode_dataset(train_descriptors)
    y_train = np.array(train_labels)
    
    # Encode test set
    X_test = bow.encode_dataset(test_descriptors)
    y_test = np.array(test_labels)
    
    print(f"\n✓ Training features: {X_train.shape}")
    print(f"✓ Test features: {X_test.shape}")
    
    # ========================================================================
    # STEP 5: Train Classifier
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Training Classifier")
    print("-"*80)
    print("Note: Training on 101 classes may take longer than CIFAR-10")
    
    classifier = ImageClassifier(
        classifier_type=CLASSIFIER_TYPE,
        C=10.0,
        max_iter=2000,  # More iterations for more classes
        random_state=42
    )
    
    classifier.train(X_train, y_train)
    
    # Save classifier
    classifier.save_model(results_dir / f'caltech101_{CLASSIFIER_TYPE}_classifier.pkl')
    
    # ========================================================================
    # STEP 6: Evaluate
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: Evaluation")
    print("-"*80)
    
    # Get actual number of classes in test set
    n_classes_actual = len(np.unique(y_test))
    print(f"Number of classes in test set: {n_classes_actual}")
    
    # Generate class names for actual classes
    class_names = [f"Class_{i}" for i in range(n_classes_actual)]
    
    results = classifier.evaluate(X_test, y_test, class_names=class_names)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        save_path=results_dir / 'caltech101_confusion_matrix.png'
    )
    
    # ========================================================================
    # STEP 7: Save Results
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: Saving Results")
    print("-"*80)
    
    training_time = (time.time() - start_time) / 60
    
    final_results = {
        'dataset': 'Caltech-101',
        'feature_type': FEATURE_TYPE,
        'max_features_per_image': MAX_FEATURES,
        'vocab_size': VOCAB_SIZE,
        'classifier_type': CLASSIFIER_TYPE,
        'test_accuracy': float(results['accuracy']),
        'training_time_minutes': training_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_classes': 101
    }
    
    with open(results_dir / 'caltech101_traditional_cv_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"✓ Results saved to: {results_dir}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  Dataset: Caltech-101 (101 classes)")
    print(f"  Feature Type: {FEATURE_TYPE}")
    print(f"  Max Features/Image: {MAX_FEATURES}")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Classifier: {CLASSIFIER_TYPE}")
    
    print("\nResults:")
    print(f"  Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Training Time: {training_time:.1f} minutes")
    
    print("\n💡 Analysis:")
    print("  • Caltech-101 is much harder than CIFAR-10 (101 vs 10 classes)")
    print("  • Traditional CV struggles with many classes")
    print("  • CNN achieved 46.42% on this dataset")
    print(f"  • Traditional CV achieved {results['accuracy']*100:.2f}%")
    print(f"  • Performance gap: {46.42 - results['accuracy']*100:.2f}%")
    
    print("\n✅ Next Steps:")
    print("  1. Compare all 4 models:")
    print("     - CNN on CIFAR-10: 79.67%")
    print("     - CNN on Caltech-101: 46.42%")
    print("     - Traditional CV on CIFAR-10: ~45%")
    print(f"     - Traditional CV on Caltech-101: {results['accuracy']*100:.2f}%")
    print("  2. Write comparative analysis (Part 5)")
    print("  3. Discuss why CNN > Traditional CV")
    print("  4. Analyze effect of dataset complexity")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()