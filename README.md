# COMP64301: Computer Vision Coursework
## Cognitive Robotics and Computer Vision Assignment

**Submission Deadline**: Friday, 5th December 2025, 14:00

---

## Project Overview

This project implements and compares two approaches for object recognition:

1. **Deep Learning Approach**: Convolutional Neural Networks (CNNs)
2. **Traditional Computer Vision Approach**: Bag-of-Words with Local Features

### Objectives
- Design, execute, and evaluate computer vision algorithms
- Perform systematic hyperparameter exploration
- Compare traditional CV methods with deep learning approaches
- Contextualize findings within robotics and computer vision state-of-the-art

---

## Project Structure

```
cv_coursework/
├── configs/                      # Configuration files
│   └── config.py                 # Main configuration
├── data/                         # Data directory
│   ├── raw/                      # Raw downloaded datasets
│   ├── processed/                # Preprocessed data
│   └── augmented/                # Augmented datasets
├── src/                          # Source code
│   ├── cnn/                      # CNN implementation
│   ├── traditional_cv/           # Traditional CV methods
│   ├── utils/                    # Utility functions
│   └── visualization/            # Plotting and visualization
├── models/                       # Saved models
│   ├── cnn/                      # Trained CNN models
│   └── traditional_cv/           # Traditional CV models (BoW, etc.)
├── results/                      # Experimental results
│   ├── cnn/                      # CNN results
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── logs/
│   ├── traditional_cv/           # Traditional CV results
│   │   ├── figures/
│   │   └── metrics/
│   └── comparison/               # Comparison results
├── notebooks/                    # Jupyter notebooks for experimentation
├── report/                       # Report materials
│   ├── figures/                  # Figures for report
│   └── tables/                   # Tables for report
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

### Method 1: Automated Setup (Recommended) ⭐

**Windows:**
```bash
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

OR use Python script (all platforms):
```bash
python setup.py
```

### Method 2: Manual Setup

#### 1. Extract the project
```bash
tar -xzf cv_coursework_dual_dataset.tar.gz
cd cv_coursework
```

#### 2. Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verify installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### Need Help?
Run the comprehensive venv guide:
```bash
python VENV_SETUP_GUIDE.py
```

---

## Dataset Options

### Datasets Implemented in This Project

**PRIMARY: CIFAR-10** ✅
- 10 classes, 60,000 images (32×32 RGB)
- Perfect for quick experimentation
- Built-in PyTorch support
- Good balance for coursework

**SECONDARY: Caltech-101** ✅
- 101 classes, ~9,000 images (variable resolution, resized to 224×224)
- Higher resolution, more realistic
- Excellent for Traditional CV methods
- Tests model generalization

### Why These Two?
- ✅ Different resolutions (32×32 vs 224×224)
- ✅ Different complexity (10 vs 101 classes)
- ✅ Different distributions (balanced vs imbalanced)
- ✅ Both suitable for CNN and Traditional CV
- ✅ Manageable training time

---

## Usage

### Training on CIFAR-10

```python
# Train baseline model on CIFAR-10
python main_cnn.py
```

### Training on Caltech-101

```python
# Train model on Caltech-101
python main_cnn_caltech101.py
```

### Train on Both Datasets (Recommended)

```python
# Train on both datasets sequentially
python train_all_datasets.py
```

### Hyperparameter Tuning

```python
# Systematic hyperparameter exploration
python hyperparameter_tuning.py
```

---

## Marking Criteria (Total: 50 marks)

| Component | Marks | Description |
|-----------|-------|-------------|
| CNN Methods | 8 | Network complexity, hyperparameters, implementation |
| CNN Results | 8 | Results presentation, interpretation, assessment |
| Traditional CV Methods | 8 | Local features implementation and justification |
| Traditional CV Results | 8 | Results presentation and analysis |
| Comparison | 3 | Deep Learning vs Traditional CV comparison |
| State of the Art | 10 | Literature review, critical analysis |
| Exceptional Performance | 5 | Bonus marks for complexity and quality |

---

## Report Requirements

- **Maximum 7 pages** (including figures, tables, references)
- **Appendix**: Code for Parts 3 & 4 (not counted in page limit)
- No cover page required
- Double-column format allowed
- Sections: Introduction, Methods, Results, Comparison, State-of-the-Art, Conclusion

---

## Important Notes

1. **Code Submission**: Only include code for Traditional CV (Parts 3 & 4) in appendix
2. **Justification**: All design choices must be justified
3. **Understanding**: Focus on demonstrating understanding, not just results
4. **Reproducibility**: Use random seeds for reproducible experiments
5. **Documentation**: Comment your code clearly

---

## Key Considerations

- RGB vs Grayscale: Must be justified if different between approaches
- Pretrained models: Allowed (e.g., ResNet-18) with fine-tuning
- Input resolution: Can vary (e.g., 64×64 to 224×224) if justified
- Hyperparameter tuning: Focus on key parameters, not exhaustive search
- Visualization: Use appropriate charts and tables

---

## Timeline Suggestions

- **Week 1**: Dataset selection, project setup, baseline CNN
- **Week 2**: CNN hyperparameter exploration
- **Week 3**: Traditional CV implementation
- **Week 4**: Comparison, state-of-the-art review, report writing

---

## Resources

- Course materials on Canvas
- FAQ document (FAQs_for_COMP64301_assignment)
- Discussion forum for questions
- Office hours with TAs

---

## Contact

For questions, use the discussion forum on Canvas or contact TAs:
- George Bird
- Kai Cao

---

## License

This project is for educational purposes as part of COMP64301 coursework.
