@echo off
REM Quick Setup Script for Windows
REM COMP64301: Computer Vision Coursework

echo ================================================================================
echo                     QUICK SETUP (Windows)                                
echo ================================================================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python not found!
    echo   Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo √ Python found
echo.

REM Create venv
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo X Failed to create virtual environment
    pause
    exit /b 1
)
echo √ Virtual environment created
echo.

REM Activate and upgrade pip
echo Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo √ pip upgraded
echo.

REM Install requirements
echo Installing dependencies (this may take 2-5 minutes)...
echo Go make some coffee!
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo X Failed to install dependencies
    pause
    exit /b 1
)
echo √ Dependencies installed
echo.

REM Verify
echo Verifying installation...
python -c "import torch; print('√ PyTorch:', torch.__version__)"
python -c "import cv2; print('√ OpenCV:', cv2.__version__)"
echo.

REM Deactivate
call deactivate

echo ================================================================================
echo                           SETUP COMPLETE!
echo ================================================================================
echo.
echo To activate your virtual environment:
echo   venv\Scripts\activate
echo.
echo Then run:
echo   python main_cnn.py
echo.
echo To deactivate when done:
echo   deactivate
echo.
echo Happy coding!
pause
