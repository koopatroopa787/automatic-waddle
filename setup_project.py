"""
Project Structure Setup Script
COMP64301: Computer Vision Coursework
"""

import os
from pathlib import Path

def create_project_structure():
    """Create organized directory structure for the coursework project"""
    
    base_dir = Path(__file__).parent
    
    directories = [
        # Data directories
        'data/raw',
        'data/processed',
        'data/augmented',
        
        # Source code directories
        'src/cnn',
        'src/traditional_cv',
        'src/utils',
        'src/visualization',
        
        # Model directories
        'models/cnn',
        'models/traditional_cv',
        
        # Results directories
        'results/cnn/figures',
        'results/cnn/metrics',
        'results/cnn/logs',
        'results/traditional_cv/figures',
        'results/traditional_cv/metrics',
        'results/comparison',
        
        # Notebooks for experimentation
        'notebooks',
        
        # Report directory
        'report/figures',
        'report/tables',
        
        # Configuration files
        'configs',
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\nProject structure created successfully!")
    return base_dir

if __name__ == "__main__":
    create_project_structure()
