#!/bin/bash
# Quick Setup Script for macOS/Linux
# COMP64301: Computer Vision Coursework

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     QUICK SETUP (macOS/Linux)                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found!"
    echo "  Please install Python 3.8 or higher"
    exit 1
fi

python3 --version
echo "✓ Python found"
echo ""

# Create venv
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate and upgrade pip
echo "Upgrading pip..."
source venv/bin/activate
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies (this may take 2-5 minutes)..."
echo "☕ Go make some coffee!"
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify
echo "Verifying installation..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
echo ""

# Deactivate
deactivate

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           SETUP COMPLETE! 🎉                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate your virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then run:"
echo "  python main_cnn.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Happy coding! 🚀"
