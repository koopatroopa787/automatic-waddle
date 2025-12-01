#!/usr/bin/env python3
"""
Automated Setup Script
COMP64301: Computer Vision Coursework

This script automates the virtual environment setup and dependency installation.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(" "*((80-len(text))//2) + text)
    print("="*80 + "\n")


def print_step(step_num, text):
    """Print step information"""
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}: {text}")
    print(f"{'─'*80}")


def run_command(cmd, description, check=True):
    """Run a shell command with nice output"""
    print(f"\n► Running: {description}")
    print(f"  Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"  ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        if e.stderr:
            print(f"  {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required!")
        print("  Please install Python 3.8+ from python.org")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_venv_exists():
    """Check if venv already exists"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠ Virtual environment already exists!")
        response = input("Delete and recreate? (y/n): ").lower()
        if response == 'y':
            print("Removing existing venv...")
            import shutil
            shutil.rmtree(venv_path)
            print("✓ Removed")
            return False
        else:
            print("Using existing venv")
            return True
    return False


def create_venv():
    """Create virtual environment"""
    print_step(1, "Creating Virtual Environment")
    
    if check_venv_exists():
        return True
    
    python_cmd = sys.executable
    
    success = run_command(
        [python_cmd, "-m", "venv", "venv"],
        "Creating venv folder"
    )
    
    if success:
        print("\n✓ Virtual environment created successfully!")
        print("  Location: ./venv")
    
    return success


def get_pip_path():
    """Get path to pip in venv"""
    system = platform.system()
    
    if system == "Windows":
        return Path("venv/Scripts/pip.exe")
    else:  # macOS, Linux
        return Path("venv/bin/pip")


def get_python_path():
    """Get path to python in venv"""
    system = platform.system()
    
    if system == "Windows":
        return Path("venv/Scripts/python.exe")
    else:  # macOS, Linux
        return Path("venv/bin/python")


def upgrade_pip():
    """Upgrade pip to latest version"""
    print_step(2, "Upgrading pip")
    
    pip_path = get_pip_path()
    
    success = run_command(
        [str(pip_path), "install", "--upgrade", "pip"],
        "Upgrading pip to latest version"
    )
    
    if success:
        print("\n✓ pip upgraded successfully!")
    
    return success


def install_requirements():
    """Install dependencies from requirements.txt"""
    print_step(3, "Installing Dependencies")
    
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found!")
        return False
    
    pip_path = get_pip_path()
    
    print("This may take 2-5 minutes (PyTorch is ~2GB)...")
    print("☕ Go make some coffee!\n")
    
    success = run_command(
        [str(pip_path), "install", "-r", "requirements.txt"],
        "Installing all required packages"
    )
    
    if success:
        print("\n✓ All dependencies installed successfully!")
    
    return success


def verify_installation():
    """Verify that key packages are installed"""
    print_step(4, "Verifying Installation")
    
    python_path = get_python_path()
    
    packages_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]
    
    all_good = True
    
    for package, name in packages_to_test:
        result = subprocess.run(
            [str(python_path), "-c", f"import {package}; print({package}.__version__)"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ {name:20s} {version}")
        else:
            print(f"  ✗ {name:20s} FAILED")
            all_good = False
    
    if all_good:
        print("\n✓ All packages verified successfully!")
    else:
        print("\n⚠ Some packages failed to import")
    
    return all_good


def print_activation_instructions():
    """Print instructions for activating venv"""
    print_step(5, "Activation Instructions")
    
    system = platform.system()
    
    print("Your virtual environment is ready! 🎉\n")
    print("To ACTIVATE the environment:\n")
    
    if system == "Windows":
        print("  Command Prompt:")
        print("    venv\\Scripts\\activate.bat\n")
        print("  PowerShell:")
        print("    venv\\Scripts\\Activate.ps1\n")
        print("  Note: If PowerShell gives an error, run:")
        print("    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser\n")
    else:  # macOS, Linux
        print("  source venv/bin/activate\n")
    
    print("To DEACTIVATE (when done):")
    print("  deactivate\n")
    
    print("Next time you work on this project:")
    print("  1. Navigate to this folder")
    print("  2. Activate the venv (see above)")
    print("  3. Start coding!")


def print_next_steps():
    """Print what to do next"""
    print_header("SETUP COMPLETE! 🎉")
    
    print("You can now run your scripts!\n")
    
    print("Quick Start:")
    print("  1. Activate venv (see instructions above)")
    print("  2. Run: python main_cnn.py\n")
    
    print("Or train on both datasets:")
    print("  python train_all_datasets.py\n")
    
    print("For detailed guides:")
    print("  python VENV_SETUP_GUIDE.py         # Virtual environment guide")
    print("  python MULTI_DATASET_GUIDE.py      # Multi-dataset training")
    print("  python CNN_USAGE_GUIDE.py          # CNN detailed usage")
    print("  python QUICKSTART.py               # Project overview\n")
    
    print("Happy coding! 🚀")


def main():
    """Main setup function"""
    print_header("AUTOMATED SETUP SCRIPT")
    print("This script will set up your virtual environment and install all dependencies.\n")
    
    # Check we're in the right directory
    if not Path("requirements.txt").exists():
        print("✗ Error: requirements.txt not found!")
        print("  Please run this script from the cv_coursework directory")
        sys.exit(1)
    
    # Check Python version
    print_step(0, "Checking Python Version")
    if not check_python_version():
        sys.exit(1)
    
    print("\nPlatform:", platform.system())
    print("Python:", sys.executable)
    
    # Create venv
    if not create_venv():
        print("\n✗ Failed to create virtual environment")
        sys.exit(1)
    
    # Upgrade pip
    if not upgrade_pip():
        print("\n⚠ Warning: pip upgrade failed (continuing anyway)")
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠ Warning: Some packages failed verification")
        print("  Try running the installation manually")
    
    # Print instructions
    print_activation_instructions()
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
