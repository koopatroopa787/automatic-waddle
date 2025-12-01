import argparse
import sys
from pathlib import Path
import time
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.traditional_cv.feature_extraction import FeatureExtractor, create_cifar10_raw_dataset, create_caltech101_raw_dataset
from src.traditional_cv.bag_of_words import BagOfWords
from src.traditional_cv.classification import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description="Train Traditional CV Pipeline")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'caltech101'])
    parser.add_argument('--feature', type=str, default='SIFT', choices=['SIFT', 'ORB'])
    parser.add_argument('--vocab_size', type=int, default=500)
    args = parser.parse_args()

    print(f"\n📸 TRADITIONAL CV: {args.feature} + BoW({args.vocab_size}) on {args.dataset.upper()}")
    
    # 1. Load Data
    if args.dataset == 'cifar10':
        train_data, test_data = create_cifar10_raw_dataset()
        datasets = {'train': train_data, 'test': test_data}
    else:
        full_data = create_caltech101_raw_dataset()
        all_indices = np.arange(len(full_data))
        train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        datasets = {
            'train': torch.utils.data.Subset(full_data, train_idx),
            'test': torch.utils.data.Subset(full_data, test_idx)
        }

    # 2. Extract Features
    extractor = FeatureExtractor(args.feature, max_features=200)
    print("Extracting features (this may take a while)...")
    
    train_descs, train_labels = extractor.extract_from_dataset(datasets['train'])
    test_descs, test_labels = extractor.extract_from_dataset(datasets['test'])

    # 3. Build Vocabulary
    bow = BagOfWords(vocab_size=args.vocab_size)
    bow.build_vocabulary(train_descs, max_samples=100000)

    # 4. Encode
    X_train = bow.encode_dataset(train_descs)
    X_test = bow.encode_dataset(test_descs)
    
    # 5. Train Classifier
    classifier = ImageClassifier('LinearSVM', C=1.0)
    classifier.train(X_train, np.array(train_labels))
    
    # 6. Evaluate
    results = classifier.evaluate(X_test, np.array(test_labels))
    
    # Save Results
    res_path = Path(f"results/traditional_cv/{args.dataset}_{args.feature}_results.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'dataset': args.dataset,
        'accuracy': results['accuracy'],
        'config': vars(args)
    }
    with open(res_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n✅ Finished! Accuracy: {results['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()