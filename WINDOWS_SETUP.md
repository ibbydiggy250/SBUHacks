# Windows Installation Guide

If you're getting "pip is not recognized" error, follow these steps:

## Option 1: Use Python's built-in pip (Recommended)

Instead of using `pip` directly, use `python -m pip`:

```bash
python -m pip install -r requirements.txt
```

Or install packages individually:
```bash
python -m pip install opencv-python
python -m pip install mediapipe
python -m pip install numpy
python -m pip install scipy
python -m pip install requests
python -m pip install streamlit
python -m pip install Pillow
```

## Option 2: Run the Installation Scripts

### Using Batch File (Command Prompt)
1. Open Command Prompt (cmd)
2. Navigate to the SBUHacks folder
3. Run: `install_windows.bat`

### Using PowerShell
1. Open PowerShell
2. Navigate to the SBUHacks folder
3. Run: `.\install_windows.ps1`

If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Option 3: Fix Python PATH

If Python is not recognized either:

1. **Download Python**: 
   - Go to https://www.python.org/downloads/
   - Download Python 3.8 or higher
   - **IMPORTANT**: During installation, check "Add Python to PATH"

2. **Manual PATH Setup** (if Python is installed but not in PATH):
   - Find Python installation (usually `C:\Python3x\` or `C:\Users\YourName\AppData\Local\Programs\Python\Python3x\`)
   - Add to PATH:
     - Right-click "This PC" → Properties
     - Advanced system settings → Environment Variables
     - Edit "Path" variable
     - Add: `C:\Python3x\` and `C:\Python3x\Scripts\`
   - Restart Command Prompt/PowerShell

## Option 4: Use Python Launcher (py)

Windows often has a Python launcher:

```bash
py -m pip install -r requirements.txt
```

Or:
```bash
py -3 -m pip install -r requirements.txt
```

## Verify Installation

After installation, verify everything works:

```bash
python --version
python -m pip --version
python -c "import cv2; import mediapipe; import numpy; print('All packages imported successfully!')"
```

## Troubleshooting

### "Python is not recognized"
- Python is not installed or not in PATH
- Reinstall Python with "Add to PATH" checked
- Or use the full path: `C:\Python3x\python.exe -m pip install ...`

### "pip is not recognized" but Python works
- Use `python -m pip` instead of `pip`
- Or install pip: `python -m ensurepip --upgrade`

### Permission errors
- Run Command Prompt/PowerShell as Administrator
- Or use user installation: `python -m pip install --user package_name`

### SSL/Certificate errors
- Update pip: `python -m pip install --upgrade pip`
- Or use trusted hosts: `python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name`

## Alternative: Use Anaconda/Miniconda

If you have Anaconda or Miniconda installed:

```bash
conda create -n asl_translator python=3.9
conda activate asl_translator
pip install -r requirements.txt
```

## Quick Test

After installation, test the setup:

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe; print('MediaPipe installed')"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

If all commands work, you're ready to run the ASL translator!

