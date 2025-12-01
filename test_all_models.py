"""
Comprehensive Model Testing with Results Saving
COMP64301: Computer Vision Coursework

Test all 4 models and save results as JSON + visualizations
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.cnn.models import BaselineCNN, ImprovedCNN
from src.traditional_cv.feature_extraction import FeatureExtractor
from src.traditional_cv.bag_of_words import BagOfWords
from src.traditional_cv.classification import ImageClassifier


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_cnn_cifar10():
    """Load trained CNN model for CIFAR-10"""
    print("\n" + "="*80)
    print("Loading CNN Model for CIFAR-10")
    print("="*80)
    
    results_dir = Path('results/cnn')
    model_dirs = list(results_dir.glob('baseline_cnn_cifar10_*'))
    
    if not model_dirs:
        print("✗ No CIFAR-10 CNN model found!")
        return None, None
    
    model_dir = model_dirs[0]
    print(f"Found model: {model_dir.name}")
    
    with open(model_dir / 'final_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"✓ Test Accuracy: {results['test_acc']:.2f}%")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineCNN(num_classes=10)
    
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return model, transform


def load_cnn_caltech101():
    """Load trained CNN model for Caltech-101"""
    print("\n" + "="*80)
    print("Loading CNN Model for Caltech-101")
    print("="*80)
    
    results_dir = Path('results/cnn')
    model_dirs = list(results_dir.glob('improved_cnn_caltech101_*'))
    
    if not model_dirs:
        print("✗ No Caltech-101 CNN model found!")
        return None, None
    
    model_dir = sorted(model_dirs)[-1]
    print(f"Found model: {model_dir.name}")
    
    with open(model_dir / 'final_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"✓ Best Val Accuracy: {results['best_val_acc']:.2f}%")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(num_classes=101)
    
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return model, transform


def load_traditional_cv_cifar10():
    """Load Traditional CV model for CIFAR-10"""
    print("\n" + "="*80)
    print("Loading Traditional CV Model for CIFAR-10")
    print("="*80)
    
    results_dir = Path('results/traditional_cv')
    
    with open(results_dir / 'cifar10_traditional_cv_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"✓ Test Accuracy: {results['test_accuracy']*100:.2f}%")
    
    bow = BagOfWords.load_model(results_dir / 'cifar10_bow_500.pkl')
    classifier = ImageClassifier.load_model(results_dir / 'cifar10_LinearSVM_classifier.pkl')
    feature_extractor = FeatureExtractor('SIFT', max_features=200)
    
    print(f"✓ Models loaded")
    
    return feature_extractor, bow, classifier


def load_traditional_cv_caltech101():
    """Load Traditional CV model for Caltech-101"""
    print("\n" + "="*80)
    print("Loading Traditional CV Model for Caltech-101")
    print("="*80)
    
    results_dir = Path('results/traditional_cv')
    
    with open(results_dir / 'caltech101_traditional_cv_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"✓ Test Accuracy: {results['test_accuracy']*100:.2f}%")
    
    bow = BagOfWords.load_model(results_dir / 'caltech101_bow_500.pkl')
    classifier = ImageClassifier.load_model(results_dir / 'caltech101_LinearSVM_classifier.pkl')
    feature_extractor = FeatureExtractor('SIFT', max_features=200)
    
    print(f"✓ Models loaded")
    
    return feature_extractor, bow, classifier


def test_and_visualize_cnn(model, transform, dataset, class_names, model_name, 
                           n_samples=20, save_dir='results/testing'):
    """Test CNN and save results + visualization"""
    print(f"\n" + "-"*80)
    print(f"Testing {model_name} on {n_samples} samples")
    print("-"*80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Get random samples
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    predictions = []
    correct = 0
    confidences = []
    
    for idx in indices:
        image, label = dataset[idx]
        
        # Predict
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                input_tensor = image.unsqueeze(0).to(device)
            else:
                input_tensor = transform(image).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred = output.argmax(dim=1).item()
            confidence = probabilities[pred].item()
        
        true_class = class_names[label] if label < len(class_names) else f"Class_{label}"
        pred_class = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        
        is_correct = pred == label
        correct += is_correct
        confidences.append(confidence)
        
        predictions.append({
            'image_idx': int(idx),
            'true_label': int(label),
            'true_class': true_class,
            'predicted_label': int(pred),
            'predicted_class': pred_class,
            'confidence': float(confidence),
            'is_correct': bool(is_correct)
        })
        
        symbol = "✓" if is_correct else "✗"
        print(f"{symbol} Image {idx:5d}: True={true_class:15s} | Pred={pred_class:15s} | Conf={confidence*100:5.1f}%")
    
    accuracy = (correct / n_samples) * 100
    avg_confidence = np.mean(confidences) * 100
    
    # Save results JSON
    results = {
        'model_name': model_name,
        'dataset': dataset.__class__.__name__,
        'n_samples_tested': n_samples,
        'accuracy': float(accuracy),
        'average_confidence': float(avg_confidence),
        'correct_predictions': int(correct),
        'incorrect_predictions': int(n_samples - correct),
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    json_path = save_dir / f'{model_name.lower().replace(" ", "_")}_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {json_path}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Avg Confidence: {avg_confidence:.2f}%")
    
    # Create visualization (6 samples)
    visualize_cnn_predictions(model, transform, dataset, class_names, 
                             model_name, save_dir, n_viz=6)
    
    return results


def visualize_cnn_predictions(model, transform, dataset, class_names, 
                              model_name, save_dir, n_viz=6):
    """Create visualization of CNN predictions"""
    device = next(model.parameters()).device
    indices = np.random.choice(len(dataset), n_viz, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Predict
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                display_image = image.numpy().transpose(1, 2, 0)
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                display_image = display_image * std + mean
                display_image = np.clip(display_image, 0, 1)
                input_tensor = image.unsqueeze(0).to(device)
            else:
                display_image = np.array(image)
                if display_image.max() > 1:
                    display_image = display_image / 255.0
                input_tensor = transform(image).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred = output.argmax(dim=1).item()
            confidence = probabilities[pred].item() * 100
        
        # Display
        axes[i].imshow(display_image)
        axes[i].axis('off')
        
        true_class = class_names[label] if label < len(class_names) else f"Class_{label}"
        pred_class = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        
        is_correct = pred == label
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()


def test_and_save_traditional_cv(feature_extractor, bow, classifier, dataset, 
                                class_names, model_name, n_samples=20, 
                                save_dir='results/testing'):
    """Test Traditional CV and save results"""
    print(f"\n" + "-"*80)
    print(f"Testing {model_name} on {n_samples} samples")
    print("-"*80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    predictions = []
    correct = 0
    feature_counts = []
    
    for idx in indices:
        image, label = dataset[idx]
        
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
        else:
            image = np.array(image)
        
        # Extract features
        descriptors = feature_extractor.extract_from_image(image)
        n_features = len(descriptors)
        feature_counts.append(n_features)
        
        # Encode and predict
        histogram = bow.encode_image(descriptors)
        pred = classifier.predict(histogram.reshape(1, -1))[0]
        
        true_class = class_names[label] if label < len(class_names) else f"Class_{label}"
        pred_class = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        
        is_correct = pred == label
        correct += is_correct
        
        predictions.append({
            'image_idx': int(idx),
            'true_label': int(label),
            'true_class': true_class,
            'predicted_label': int(pred),
            'predicted_class': pred_class,
            'n_features': int(n_features),
            'is_correct': bool(is_correct)
        })
        
        symbol = "✓" if is_correct else "✗"
        print(f"{symbol} Image {idx:5d}: True={true_class:15s} | Pred={pred_class:15s} | Features={n_features:3d}")
    
    accuracy = (correct / n_samples) * 100
    avg_features = np.mean(feature_counts)
    
    # Save results JSON
    results = {
        'model_name': model_name,
        'dataset': 'CIFAR10' if 'CIFAR' in model_name else 'Caltech101',
        'n_samples_tested': n_samples,
        'accuracy': float(accuracy),
        'average_features': float(avg_features),
        'min_features': int(np.min(feature_counts)),
        'max_features': int(np.max(feature_counts)),
        'correct_predictions': int(correct),
        'incorrect_predictions': int(n_samples - correct),
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    json_path = save_dir / f'{model_name.lower().replace(" ", "_")}_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {json_path}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Avg Features: {avg_features:.1f}")
    
    return results


def create_comparison_summary(all_results, save_dir):
    """Create comprehensive comparison of all models"""
    save_dir = Path(save_dir)
    
    summary = {
        'comparison_summary': {
            'timestamp': datetime.now().isoformat(),
            'models_tested': len(all_results),
            'models': {}
        }
    }
    
    for result in all_results:
        model_name = result['model_name']
        summary['comparison_summary']['models'][model_name] = {
            'dataset': result['dataset'],
            'accuracy': result['accuracy'],
            'samples_tested': result['n_samples_tested'],
            'correct': result['correct_predictions'],
            'incorrect': result['incorrect_predictions']
        }
        
        if 'average_confidence' in result:
            summary['comparison_summary']['models'][model_name]['avg_confidence'] = result['average_confidence']
        if 'average_features' in result:
            summary['comparison_summary']['models'][model_name]['avg_features'] = result['average_features']
    
    # Save summary
    summary_path = save_dir / 'all_models_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Comparison summary saved to: {summary_path}")
    
    return summary


def main():
    """Main testing function"""
    print("\n" + "="*80)
    print(" "*15 + "COMPREHENSIVE MODEL TESTING WITH SAVING")
    print("="*80)
    
    save_dir = Path('results/testing')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # ========================================================================
    # Test CNN on CIFAR-10
    # ========================================================================
    try:
        model_cnn_cifar, transform_cifar = load_cnn_cifar10()
        
        if model_cnn_cifar is not None:
            test_dataset = datasets.CIFAR10(
                root='./data/raw',
                train=False,
                download=True,
                transform=transform_cifar
            )
            
            results = test_and_visualize_cnn(
                model_cnn_cifar, transform_cifar, test_dataset, 
                CIFAR10_CLASSES, 'CNN CIFAR-10', n_samples=50, save_dir=save_dir
            )
            all_results.append(results)
    except Exception as e:
        print(f"✗ Error testing CNN CIFAR-10: {e}")
    
    # ========================================================================
    # Test CNN on Caltech-101
    # ========================================================================
    try:
        model_cnn_caltech, transform_caltech = load_cnn_caltech101()
        
        if model_cnn_caltech is not None:
            full_dataset = datasets.Caltech101(
                root='./data/raw',
                download=True,
                transform=transform_caltech
            )
            
            test_indices = [i for i, (_, label) in enumerate(full_dataset) if label != 0]
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            
            caltech_classes = [f"Class_{i}" for i in range(101)]
            
            results = test_and_visualize_cnn(
                model_cnn_caltech, transform_caltech, test_dataset,
                caltech_classes, 'CNN Caltech-101', n_samples=50, save_dir=save_dir
            )
            all_results.append(results)
    except Exception as e:
        print(f"✗ Error testing CNN Caltech-101: {e}")
    
    # ========================================================================
    # Test Traditional CV on CIFAR-10
    # ========================================================================
    try:
        feat_ext_cifar, bow_cifar, clf_cifar = load_traditional_cv_cifar10()
        
        from src.traditional_cv.feature_extraction import create_cifar10_raw_dataset
        _, test_dataset_raw = create_cifar10_raw_dataset()
        
        results = test_and_save_traditional_cv(
            feat_ext_cifar, bow_cifar, clf_cifar,
            test_dataset_raw, CIFAR10_CLASSES, 'Traditional CV CIFAR-10',
            n_samples=50, save_dir=save_dir
        )
        all_results.append(results)
    except Exception as e:
        print(f"✗ Error testing Traditional CV CIFAR-10: {e}")
    
    # ========================================================================
    # Test Traditional CV on Caltech-101
    # ========================================================================
    try:
        feat_ext_caltech, bow_caltech, clf_caltech = load_traditional_cv_caltech101()
        
        from src.traditional_cv.feature_extraction import create_caltech101_raw_dataset
        test_dataset_raw = create_caltech101_raw_dataset()
        
        caltech_classes = [f"Class_{i}" for i in range(101)]
        
        results = test_and_save_traditional_cv(
            feat_ext_caltech, bow_caltech, clf_caltech,
            test_dataset_raw, caltech_classes, 'Traditional CV Caltech-101',
            n_samples=50, save_dir=save_dir
        )
        all_results.append(results)
    except Exception as e:
        print(f"✗ Error testing Traditional CV Caltech-101: {e}")
    
    # ========================================================================
    # Create Comparison Summary
    # ========================================================================
    if all_results:
        summary = create_comparison_summary(all_results, save_dir)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "TESTING COMPLETE!")
    print("="*80)
    
    print(f"\n📁 All results saved to: {save_dir.absolute()}")
    print("\n📊 Generated Files:")
    print("   • cnn_cifar-10_test_results.json")
    print("   • cnn_cifar-10_predictions.png")
    print("   • cnn_caltech-101_test_results.json")
    print("   • cnn_caltech-101_predictions.png")
    print("   • traditional_cv_cifar-10_test_results.json")
    print("   • traditional_cv_caltech-101_test_results.json")
    print("   • all_models_summary.json")
    
    print("\n💡 Use these files in your report:")
    print("   • JSON files: Extract accuracy, confidence, feature counts")
    print("   • PNG files: Include as figures showing predictions")
    print("   • Summary JSON: Create comparison tables")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()