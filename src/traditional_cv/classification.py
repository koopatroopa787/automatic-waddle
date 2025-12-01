"""
Traditional Computer Vision - Classification
COMP64301: Computer Vision Coursework

Implements SVM and Random Forest classifiers for Bag-of-Words features
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ImageClassifier:
    """
    Wrapper for scikit-learn classifiers
    """
    
    def __init__(self, classifier_type='SVM', **kwargs):
        """
        Initialize classifier
        
        Args:
            classifier_type: 'SVM', 'LinearSVM', or 'RandomForest'
            **kwargs: Additional arguments for the classifier
        """
        self.classifier_type = classifier_type
        
        if classifier_type == 'SVM':
            # RBF kernel SVM
            self.classifier = SVC(
                kernel='rbf',
                C=kwargs.get('C', 10.0),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42),
                verbose=True
            )
        elif classifier_type == 'LinearSVM':
            # Linear SVM (faster for large datasets)
            self.classifier = LinearSVC(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                verbose=1
            )
        elif classifier_type == 'RandomForest':
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),
                verbose=1
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        print(f"Initialized {classifier_type} classifier")
    
    def train(self, X_train, y_train):
        """
        Train the classifier
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
        """
        print(f"\nTraining {self.classifier_type}...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Number of classes: {len(np.unique(y_train))}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        self.classifier.fit(X_train, y_train)
        
        print("✓ Training complete!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            predictions: Predicted labels (n_samples,)
        """
        return self.classifier.predict(X)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate classifier on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Optional list of class names
            
        Returns:
            results: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {self.classifier_type}...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Test Accuracy: {accuracy * 100:.2f}%")
        
        # Classification report
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
        
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names, 
                                      zero_division=0)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'{self.classifier_type} - Confusion Matrix', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def save_model(self, save_path):
        """Save trained classifier"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'classifier_type': self.classifier_type,
            'classifier': self.classifier
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Classifier saved to: {save_path}")
    
    @staticmethod
    def load_model(load_path):
        """Load saved classifier"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        clf = ImageClassifier(classifier_type=data['classifier_type'])
        clf.classifier = data['classifier']
        
        print(f"Loaded {clf.classifier_type} classifier")
        
        return clf


if __name__ == "__main__":
    print("Testing Classifiers...")
    
    # Generate test data
    print("\nGenerating test data...")
    n_samples = 1000
    n_features = 500
    n_classes = 10
    
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_samples)
    
    X_test = np.random.randn(200, n_features).astype(np.float32)
    y_test = np.random.randint(0, n_classes, 200)
    
    # Test SVM
    print("\n" + "="*80)
    print("Testing Linear SVM")
    print("="*80)
    
    svm = ImageClassifier('LinearSVM', C=1.0, max_iter=100)
    svm.train(X_train, y_train)
    results = svm.evaluate(X_test, y_test)
    
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    
    # Test Random Forest
    print("\n" + "="*80)
    print("Testing Random Forest")
    print("="*80)
    
    rf = ImageClassifier('RandomForest', n_estimators=50)
    rf.train(X_train, y_train)
    results = rf.evaluate(X_test, y_test)
    
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    
    print("\n✓ Classifier test complete!")
