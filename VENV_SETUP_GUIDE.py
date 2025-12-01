"""
Virtual Environment Setup Guide
COMP64301: Computer Vision Coursework

Step-by-step instructions for setting up a Python virtual environment
"""

VENV_SETUP_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VIRTUAL ENVIRONMENT SETUP GUIDE                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

A virtual environment (venv) keeps your project dependencies isolated and clean.
This guide covers setup on Windows, macOS, and Linux.

════════════════════════════════════════════════════════════════════════════════
📦 WHAT IS A VIRTUAL ENVIRONMENT?
════════════════════════════════════════════════════════════════════════════════

A venv is an isolated Python environment that:
  ✓ Keeps project dependencies separate from system Python
  ✓ Prevents version conflicts between projects
  ✓ Makes your project reproducible
  ✓ Allows easy sharing with others

Think of it as a "bubble" for your project's packages.

════════════════════════════════════════════════════════════════════════════════
🚀 QUICK START (3 STEPS)
════════════════════════════════════════════════════════════════════════════════

After extracting your project:

1. CREATE the virtual environment
2. ACTIVATE it
3. INSTALL dependencies

Let's go through each step for your operating system...

════════════════════════════════════════════════════════════════════════════════
🪟 WINDOWS SETUP
════════════════════════════════════════════════════════════════════════════════

STEP 1: Extract and Navigate
─────────────────────────────
Open Command Prompt or PowerShell:

    cd path\to\extracted\folder
    cd cv_coursework

Example:
    cd C:\Users\YourName\Downloads\cv_coursework


STEP 2: Create Virtual Environment
───────────────────────────────────
Run this command:

    python -m venv venv

What this does:
  • Creates a folder called "venv" 
  • Installs a clean Python environment inside it
  • Takes about 10-30 seconds

If you get an error:
  • Try: python3 -m venv venv
  • Or: py -m venv venv


STEP 3: Activate Virtual Environment
─────────────────────────────────────

For Command Prompt (cmd):
    venv\Scripts\activate.bat

For PowerShell:
    venv\Scripts\Activate.ps1

⚠️ PowerShell Users: If you get "execution policy" error:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    Then try activating again.

You'll know it worked when you see (venv) at the start of your prompt:
    (venv) C:\Users\YourName\cv_coursework>


STEP 4: Install Dependencies
─────────────────────────────
With the venv activated:

    pip install -r requirements.txt

This installs all required packages (PyTorch, OpenCV, etc.)
Takes 2-5 minutes depending on internet speed.


STEP 5: Verify Installation
────────────────────────────
Test that everything works:

    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import cv2; print('OpenCV:', cv2.__version__)"

You should see version numbers (no errors).


STEP 6: Run Your First Script
──────────────────────────────
    python main_cnn.py

Success! Your environment is ready! ✓


TO DEACTIVATE (when done):
───────────────────────────
    deactivate


NEXT TIME YOU WORK:
───────────────────────────
1. Navigate to project: cd cv_coursework
2. Activate venv: venv\Scripts\activate
3. Start working!


════════════════════════════════════════════════════════════════════════════════
🍎 macOS SETUP
════════════════════════════════════════════════════════════════════════════════

STEP 1: Extract and Navigate
─────────────────────────────
Open Terminal:

    cd ~/Downloads
    tar -xzf cv_coursework_dual_dataset.tar.gz
    cd cv_coursework


STEP 2: Create Virtual Environment
───────────────────────────────────
Run this command:

    python3 -m venv venv

Note: Use python3 (not python) on macOS

What this does:
  • Creates a folder called "venv"
  • Installs a clean Python environment inside it
  • Takes about 10-30 seconds


STEP 3: Activate Virtual Environment
─────────────────────────────────────
    source venv/bin/activate

You'll know it worked when you see (venv) in your prompt:
    (venv) user@mac cv_coursework %


STEP 4: Upgrade pip (recommended)
──────────────────────────────────
    pip install --upgrade pip


STEP 5: Install Dependencies
─────────────────────────────
    pip install -r requirements.txt

This installs all required packages.
Takes 2-5 minutes depending on internet speed.

⚠️ If you have an M1/M2/M3 Mac:
PyTorch will automatically use the optimized ARM version.


STEP 6: Verify Installation
────────────────────────────
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import cv2; print('OpenCV:', cv2.__version__)"

You should see version numbers (no errors).


STEP 7: Run Your First Script
──────────────────────────────
    python main_cnn.py

Success! Your environment is ready! ✓


TO DEACTIVATE (when done):
───────────────────────────
    deactivate


NEXT TIME YOU WORK:
───────────────────────────
1. Navigate to project: cd ~/path/to/cv_coursework
2. Activate venv: source venv/bin/activate
3. Start working!


════════════════════════════════════════════════════════════════════════════════
🐧 LINUX SETUP
════════════════════════════════════════════════════════════════════════════════

STEP 1: Extract and Navigate
─────────────────────────────
Open Terminal:

    cd ~/Downloads
    tar -xzf cv_coursework_dual_dataset.tar.gz
    cd cv_coursework


STEP 2: Install Python venv (if not installed)
───────────────────────────────────────────────
On Ubuntu/Debian:
    sudo apt update
    sudo apt install python3-venv python3-pip

On Fedora:
    sudo dnf install python3-virtualenv

On Arch:
    sudo pacman -S python-virtualenv


STEP 3: Create Virtual Environment
───────────────────────────────────
    python3 -m venv venv

What this does:
  • Creates a folder called "venv"
  • Installs a clean Python environment inside it
  • Takes about 10-30 seconds


STEP 4: Activate Virtual Environment
─────────────────────────────────────
    source venv/bin/activate

You'll know it worked when you see (venv) in your prompt:
    (venv) user@linux:~/cv_coursework$


STEP 5: Upgrade pip (recommended)
──────────────────────────────────
    pip install --upgrade pip


STEP 6: Install Dependencies
─────────────────────────────
    pip install -r requirements.txt

This installs all required packages.
Takes 2-5 minutes depending on internet speed.


STEP 7: Verify Installation
────────────────────────────
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import cv2; print('OpenCV:', cv2.__version__)"

You should see version numbers (no errors).


STEP 8: Run Your First Script
──────────────────────────────
    python main_cnn.py

Success! Your environment is ready! ✓


TO DEACTIVATE (when done):
───────────────────────────
    deactivate


NEXT TIME YOU WORK:
───────────────────────────
1. Navigate to project: cd ~/path/to/cv_coursework
2. Activate venv: source venv/bin/activate
3. Start working!


════════════════════════════════════════════════════════════════════════════════
💡 COMMON ISSUES & SOLUTIONS
════════════════════════════════════════════════════════════════════════════════

ISSUE 1: "python not found" or "python3 not found"
───────────────────────────────────────────────────
Solution:
  • Windows: Install Python from python.org (3.8 or newer)
  • macOS: Install via Homebrew: brew install python3
  • Linux: sudo apt install python3 python3-pip


ISSUE 2: "pip not found"
─────────────────────────
Solution:
  • Activate venv first
  • Or install: python -m ensurepip --upgrade


ISSUE 3: PowerShell won't let me activate (Windows)
────────────────────────────────────────────────────
Error: "execution policy"
Solution:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


ISSUE 4: "No module named 'torch'" after installing
────────────────────────────────────────────────────
Solution:
  • Make sure venv is ACTIVATED (you should see (venv) in prompt)
  • Re-run: pip install -r requirements.txt
  • Verify: python -c "import torch"


ISSUE 5: Installation is very slow
───────────────────────────────────
Solution:
  • This is normal! PyTorch is a large package (~2 GB)
  • Go make coffee ☕
  • Use faster mirror (if in China):
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


ISSUE 6: Out of disk space
───────────────────────────
Solution:
  • Free up at least 5 GB
  • PyTorch + dependencies need ~3-4 GB


ISSUE 7: Conda vs venv confusion
─────────────────────────────────
If you have Anaconda/Miniconda:
  • Deactivate conda: conda deactivate
  • Then follow venv steps above
  • Or use conda instead (see below)


════════════════════════════════════════════════════════════════════════════════
🐍 ALTERNATIVE: USING CONDA (If you have Anaconda/Miniconda)
════════════════════════════════════════════════════════════════════════════════

If you prefer conda environments:

STEP 1: Create conda environment
─────────────────────────────────
    conda create -n cv_coursework python=3.10

STEP 2: Activate it
───────────────────
    conda activate cv_coursework

STEP 3: Install dependencies
─────────────────────────────
    pip install -r requirements.txt

OR install with conda (slower but more compatible):
    conda install pytorch torchvision -c pytorch
    pip install opencv-python scikit-learn matplotlib seaborn tqdm

STEP 4: Verify and run
───────────────────────
    python main_cnn.py

TO DEACTIVATE:
    conda deactivate


════════════════════════════════════════════════════════════════════════════════
📝 QUICK REFERENCE CARD
════════════════════════════════════════════════════════════════════════════════

CREATE VENV:
  Windows:  python -m venv venv
  Mac/Linux: python3 -m venv venv

ACTIVATE VENV:
  Windows CMD:     venv\Scripts\activate.bat
  Windows PS:      venv\Scripts\Activate.ps1
  Mac/Linux:       source venv/bin/activate

INSTALL PACKAGES:
  pip install -r requirements.txt

DEACTIVATE:
  deactivate

DELETE VENV (if you want to start over):
  Windows:  rmdir /s venv
  Mac/Linux: rm -rf venv


════════════════════════════════════════════════════════════════════════════════
✅ VERIFICATION CHECKLIST
════════════════════════════════════════════════════════════════════════════════

After setup, you should be able to:

□ See (venv) in your command prompt
□ Run: python --version (should show 3.8+)
□ Run: pip list (should show installed packages)
□ Import torch: python -c "import torch"
□ Import cv2: python -c "import cv2"
□ Run main script: python main_cnn.py

If ALL checks pass: ✓ You're ready to go!


════════════════════════════════════════════════════════════════════════════════
🎯 DAILY WORKFLOW
════════════════════════════════════════════════════════════════════════════════

Every time you work on the project:

1. Open Terminal/Command Prompt
2. Navigate to project: cd cv_coursework
3. Activate venv:
   • Windows: venv\Scripts\activate
   • Mac/Linux: source venv/bin/activate
4. Work on your code
5. When done: deactivate

That's it! 🎉


════════════════════════════════════════════════════════════════════════════════
💾 BEST PRACTICES
════════════════════════════════════════════════════════════════════════════════

✓ ALWAYS activate venv before working
✓ Keep requirements.txt updated if you add packages
✓ Don't commit venv folder to git (it's in .gitignore)
✓ One venv per project
✓ Deactivate when switching projects
✓ If something breaks, delete venv and recreate it


════════════════════════════════════════════════════════════════════════════════
🆘 STILL STUCK?
════════════════════════════════════════════════════════════════════════════════

1. Check Python version: python --version (need 3.8+)
2. Try creating venv in a folder without spaces in the path
3. Make sure you have administrator/sudo access
4. Google the specific error message
5. Ask on the discussion forum
6. Contact TAs: George Bird, Kai Cao


════════════════════════════════════════════════════════════════════════════════
📚 MORE RESOURCES
════════════════════════════════════════════════════════════════════════════════

Official Python venv docs:
  https://docs.python.org/3/library/venv.html

Virtual Environments tutorial:
  https://realpython.com/python-virtual-environments-a-primer/


════════════════════════════════════════════════════════════════════════════════

You're all set! Once your venv is activated, you can run any script in the 
project without worrying about dependencies. 🚀

Happy coding! 🎓
"""

if __name__ == "__main__":
    print(VENV_SETUP_GUIDE)
