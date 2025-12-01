"""
Traditional CV Training on CIFAR-10
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

from src.traditional_cv.feature_extraction import FeatureExtractor, create_cifar10_raw_dataset
from src.traditional_cv.bag_of_words import BagOfWords
from src.traditional_cv.classification import ImageClassifier
import numpy as np


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" "*15 + "TRADITIONAL CV ON CIFAR-10")
    print(" "*10 + "(SIFT + Bag-of-Words + SVM)")
    print("="*80)
    
    # Configuration
    FEATURE_TYPE = 'SIFT'  # or 'ORB'
    MAX_FEATURES = 200
    VOCAB_SIZE = 500
    CLASSIFIER_TYPE = 'LinearSVM'  # or 'SVM', 'RandomForest'
    
    # Paths
    results_dir = Path('results/traditional_cv')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading CIFAR-10 Dataset")
    print("-"*80)
    
    train_dataset, test_dataset = create_cifar10_raw_dataset()
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # ========================================================================
    # STEP 2: Extract Features
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Extracting Features")
    print("-"*80)
    
    feature_extractor = FeatureExtractor(
        feature_type=FEATURE_TYPE,
        max_features=MAX_FEATURES
    )
    
    # Extract from training set
    print("\nExtracting training features...")
    train_descriptors, train_labels = feature_extractor.extract_from_dataset(
        train_dataset,
        max_images=None  # Use all images
    )
    
    # Extract from test set
    print("\nExtracting test features...")
    test_descriptors, test_labels = feature_extractor.extract_from_dataset(
        test_dataset,
        max_images=None
    )
    
    # Save features
    feature_extractor.save_features(
        train_descriptors,
        train_labels,
        results_dir / f'cifar10_train_{FEATURE_TYPE}_features.pkl'
    )
    feature_extractor.save_features(
        test_descriptors,
        test_labels,
        results_dir / f'cifar10_test_{FEATURE_TYPE}_features.pkl'
    )
    
    # ========================================================================
    # STEP 3: Build Vocabulary (Bag-of-Words)
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Building Visual Vocabulary")
    print("-"*80)
    
    bow = BagOfWords(vocab_size=VOCAB_SIZE, random_state=42)
    
    # Use subset of descriptors for faster clustering
    bow.build_vocabulary(train_descriptors, max_samples=100000)
    
    # Save BoW model
    bow.save_model(results_dir / f'cifar10_bow_{VOCAB_SIZE}.pkl')
    
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
    
    classifier = ImageClassifier(
        classifier_type=CLASSIFIER_TYPE,
        C=10.0,
        max_iter=1000,
        random_state=42
    )
    
    classifier.train(X_train, y_train)
    
    # Save classifier
    classifier.save_model(results_dir / f'cifar10_{CLASSIFIER_TYPE}_classifier.pkl')
    
    # ========================================================================
    # STEP 6: Evaluate
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: Evaluation")
    print("-"*80)
    
    results = classifier.evaluate(X_test, y_test, class_names=CIFAR10_CLASSES)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        results['confusion_matrix'],
        CIFAR10_CLASSES,
        save_path=results_dir / 'cifar10_confusion_matrix.png'
    )
    
    # ========================================================================
    # STEP 7: Save Results
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: Saving Results")
    print("-"*80)
    
    training_time = (time.time() - start_time) / 60
    
    final_results = {
        'dataset': 'CIFAR-10',
        'feature_type': FEATURE_TYPE,
        'max_features_per_image': MAX_FEATURES,
        'vocab_size': VOCAB_SIZE,
        'classifier_type': CLASSIFIER_TYPE,
        'test_accuracy': float(results['accuracy']),
        'training_time_minutes': training_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open(results_dir / 'cifar10_traditional_cv_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"✓ Results saved to: {results_dir}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  Feature Type: {FEATURE_TYPE}")
    print(f"  Max Features/Image: {MAX_FEATURES}")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Classifier: {CLASSIFIER_TYPE}")
    
    print("\nResults:")
    print(f"  Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Training Time: {training_time:.1f} minutes")
    
    print("\nNext Steps:")
    print("  1. Run on Caltech-101: python main_traditional_cv_caltech101.py")
    print("  2. Compare with CNN results")
    print("  3. Try different parameters (vocab_size, feature_type, classifier)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
