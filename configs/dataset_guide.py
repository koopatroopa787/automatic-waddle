"""
Dataset Selection Guide
COMP64301: Computer Vision Coursework

This document helps you choose appropriate datasets for the coursework.
"""

DATASET_COMPARISON = {
    'CIFAR-10': {
        'classes': 10,
        'images': 60000,
        'resolution': '32x32',
        'color': 'RGB',
        'difficulty': 'Easy-Medium',
        'download_size': '~170 MB',
        'pros': [
            'Small and fast to download/train',
            'Well-established benchmark',
            'Built-in PyTorch support',
            'Good for initial experiments',
            'Perfect for comparing CNN vs Traditional CV'
        ],
        'cons': [
            'Low resolution (32x32)',
            'Relatively simple patterns',
            'Limited to 10 classes'
        ],
        'recommended_for': 'Quick prototyping, baseline experiments',
        'pytorch_available': True
    },
    
    'CIFAR-100': {
        'classes': 100,
        'images': 60000,
        'resolution': '32x32',
        'color': 'RGB',
        'difficulty': 'Medium',
        'download_size': '~170 MB',
        'pros': [
            'More challenging than CIFAR-10',
            'Same size as CIFAR-10 (easy comparison)',
            'Built-in PyTorch support',
            '100 diverse classes'
        ],
        'cons': [
            'Low resolution (32x32)',
            'Fewer samples per class (600 vs 6000)',
            'More difficult for traditional CV methods'
        ],
        'recommended_for': 'Testing model capacity, fine-grained classification',
        'pytorch_available': True
    },
    
    'ImageNet-subset': {
        'classes': 'Variable (often 100-1000)',
        'images': 'Variable',
        'resolution': '224x224 (varies)',
        'color': 'RGB',
        'difficulty': 'Medium-Hard',
        'download_size': 'Large (several GB)',
        'pros': [
            'High resolution images',
            'Diverse object categories',
            'Industry standard benchmark',
            'Good for transfer learning'
        ],
        'cons': [
            'Very large download',
            'Requires more compute resources',
            'Manual download needed'
        ],
        'recommended_for': 'High-quality results, pretrained models',
        'pytorch_available': 'Limited (torchvision has some subsets)'
    },
    
    'Caltech-101': {
        'classes': 101,
        'images': '~9000',
        'resolution': 'Variable (~300x200)',
        'color': 'RGB',
        'difficulty': 'Medium',
        'download_size': '~130 MB',
        'pros': [
            'Diverse object categories',
            'Variable image sizes (realistic)',
            'Good for traditional CV features',
            'Manageable size'
        ],
        'cons': [
            'Smaller dataset than CIFAR',
            'Imbalanced classes',
            'Manual download required'
        ],
        'recommended_for': 'Traditional CV experiments, feature extraction',
        'pytorch_available': False
    },
    
    'Fashion-MNIST': {
        'classes': 10,
        'images': 70000,
        'resolution': '28x28',
        'color': 'Grayscale',
        'difficulty': 'Easy',
        'download_size': '~30 MB',
        'pros': [
            'Very small and fast',
            'Easy to work with',
            'Built-in PyTorch support',
            'Good for initial testing'
        ],
        'cons': [
            'Grayscale only',
            'Low resolution',
            'Too simple for demonstrating full capabilities',
            'Limited to fashion items'
        ],
        'recommended_for': 'Rapid prototyping, code testing',
        'pytorch_available': True
    },
    
    'SVHN': {
        'classes': 10,
        'images': '~600,000',
        'resolution': '32x32',
        'color': 'RGB',
        'difficulty': 'Medium',
        'download_size': '~1.5 GB',
        'pros': [
            'Real-world images (street view)',
            'Large dataset',
            'Challenging for traditional methods',
            'Built-in torchvision support'
        ],
        'cons': [
            'Single domain (house numbers)',
            'Large download size',
            'Imbalanced classes'
        ],
        'recommended_for': 'Real-world image challenges',
        'pytorch_available': True
    },
    
    'STL-10': {
        'classes': 10,
        'images': 13000,
        'resolution': '96x96',
        'color': 'RGB',
        'difficulty': 'Medium',
        'download_size': '~2.5 GB',
        'pros': [
            'Higher resolution than CIFAR',
            'Similar classes to CIFAR-10',
            'Built-in PyTorch support',
            'Good for feature learning'
        ],
        'cons': [
            'Smaller training set (5000 images)',
            'Limited classes',
            'Large download for limited data'
        ],
        'recommended_for': 'Higher resolution experiments',
        'pytorch_available': True
    }
}


RECOMMENDATIONS = {
    'Best for Coursework (Balanced)': [
        'CIFAR-10 (primary recommendation)',
        'CIFAR-100 (if you want more challenge)',
    ],
    
    'Best for Traditional CV': [
        'CIFAR-10 (good SIFT/ORB features despite low resolution)',
        'Caltech-101 (excellent for BoW methods)',
    ],
    
    'Best for Deep Learning Showcase': [
        'CIFAR-10 or CIFAR-100',
        'STL-10 (higher resolution)',
    ],
    
    'Best for Quick Experiments': [
        'CIFAR-10',
        'Fashion-MNIST',
    ],
    
    'Multi-Dataset Strategy': {
        'description': 'Use multiple datasets to test generalization',
        'suggestion': [
            'Primary: CIFAR-10 (compare both methods)',
            'Secondary: CIFAR-100 or STL-10 (test scalability)',
        ]
    }
}


SUGGESTED_APPROACH = """
RECOMMENDED APPROACH FOR COURSEWORK:

1. PRIMARY DATASET: CIFAR-10
   - Perfect balance of complexity and manageability
   - Both methods work well (fair comparison)
   - Fast iteration and experimentation
   - Well-documented and easy to use
   
2. OPTIONAL SECONDARY: CIFAR-100 or Caltech-101
   - Shows scalability to more classes
   - Demonstrates generalization
   - Provides additional analysis points
   
3. WHY NOT OTHERS?
   - Fashion-MNIST: Too simple, grayscale only
   - ImageNet: Too large for coursework timeline
   - SVHN: Single domain, less interesting for comparison
   
4. IMPLEMENTATION PRIORITY:
   Week 1: Get CIFAR-10 working with baseline CNN
   Week 2: Hyperparameter tuning on CIFAR-10
   Week 3: Traditional CV on CIFAR-10
   Week 4: (Optional) Test on second dataset + report writing
"""


def print_dataset_info():
    """Print formatted dataset information"""
    print("\n" + "="*80)
    print("DATASET COMPARISON FOR COMP64301 COURSEWORK")
    print("="*80 + "\n")
    
    for dataset_name, info in DATASET_COMPARISON.items():
        print(f"\n{dataset_name}")
        print("-" * len(dataset_name))
        print(f"Classes: {info['classes']}")
        print(f"Images: {info['images']}")
        print(f"Resolution: {info['resolution']}")
        print(f"Color: {info['color']}")
        print(f"Difficulty: {info['difficulty']}")
        print(f"Download Size: {info['download_size']}")
        print(f"PyTorch Built-in: {'Yes' if info['pytorch_available'] else 'No'}")
        print(f"\nPros:")
        for pro in info['pros']:
            print(f"  + {pro}")
        print(f"Cons:")
        for con in info['cons']:
            print(f"  - {con}")
        print(f"Best for: {info['recommended_for']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    for category, datasets in RECOMMENDATIONS.items():
        if isinstance(datasets, dict):
            print(f"\n{category}:")
            print(f"  {datasets['description']}")
            for suggestion in datasets['suggestion']:
                print(f"  - {suggestion}")
        else:
            print(f"\n{category}:")
            for dataset in datasets:
                print(f"  - {dataset}")
    
    print("\n" + "="*80)
    print(SUGGESTED_APPROACH)
    print("="*80 + "\n")


if __name__ == "__main__":
    print_dataset_info()
