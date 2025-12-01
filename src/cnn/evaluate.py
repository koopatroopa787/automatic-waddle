"""
Model Evaluation Script
COMP64301: Computer Vision Coursework

Comprehensive evaluation of trained CNN models.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.cnn.data_loader import CIFAR10DataLoader
from src.visualization.plots import plot_confusion_matrix


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, model, test_loader, device, class_names):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self):
        """
        Perform comprehensive evaluation
        
        Returns:
            dict: Evaluation results
        """
        print("\n" + "="*80)
        print(" "*25 + "MODEL EVALUATION")
        print("="*80)
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate overall accuracy
        accuracy = 100.0 * (all_preds == all_targets).sum() / len(all_targets)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        
        # Print results
        print("\n" + "-"*80)
        print("OVERALL RESULTS")
        print("-"*80)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Total Samples: {len(all_targets)}")
        print(f"Correct Predictions: {(all_preds == all_targets).sum()}")
        
        print("\n" + "-"*80)
        print("PER-CLASS ACCURACY")
        print("-"*80)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:12s}: {per_class_acc[i]:6.2f}%  "
                  f"({cm[i, i]}/{cm[i].sum()} correct)")
        
        print("\n" + "-"*80)
        print("CLASSIFICATION REPORT")
        print("-"*80)
        print(classification_report(
            all_targets,
            all_preds,
            target_names=self.class_names,
            digits=3
        ))
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(cm, self.class_names)
        
        results = {
            'accuracy': accuracy,
            'per_class_accuracy': dict(zip(self.class_names, per_class_acc)),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probs.tolist()
        }
        
        return results
    
    def find_misclassified(self, num_samples=10):
        """
        Find misclassified samples
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            list: Misclassified samples information
        """
        misclassified = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                # Find misclassified in this batch
                mask = predicted != targets
                
                if mask.any():
                    for idx in mask.nonzero(as_tuple=True)[0]:
                        misclassified.append({
                            'image': inputs[idx].cpu(),
                            'true_label': self.class_names[targets[idx]],
                            'predicted_label': self.class_names[predicted[idx]],
                            'true_idx': targets[idx].item(),
                            'pred_idx': predicted[idx].item()
                        })
                        
                        if len(misclassified) >= num_samples:
                            return misclassified
        
        return misclassified


def evaluate_saved_model(model_path, data_loader):
    """
    Evaluate a saved model
    
    Args:
        model_path: Path to saved model
        data_loader: CIFAR10DataLoader instance
    """
    from src.cnn.models import create_model
    from src.utils.helpers import get_device
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path)
    
    # Create model architecture
    model = create_model('baseline', num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get device
    device = get_device()
    
    # Get test loader
    _, _, test_loader = data_loader.get_loaders()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=CIFAR10DataLoader.CLASS_NAMES
    )
    
    # Evaluate
    results = evaluator.evaluate()
    
    return results


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Import this module to evaluate trained models")
