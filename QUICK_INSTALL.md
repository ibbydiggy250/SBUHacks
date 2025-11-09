# Quick Install Guide

## Your Python Path
Based on your error, Python is installed at:
```
C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe
```

## Quick Fix - Install Dependencies

### Option 1: Run the Install Script (Easiest)

**In PowerShell:**
```powershell
.\install_dependencies.ps1
```

**Or in Command Prompt:**
```cmd
install_dependencies.bat
```

### Option 2: Manual Installation

Copy and paste this into PowerShell:

```powershell
$python = "C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe"
$python -m pip install --upgrade pip
$python -m pip install opencv-python mediapipe numpy scipy requests streamlit Pillow
```

### Option 3: One-Line Command

```powershell
C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe -m pip install -r requirements.txt
```

## Verify Installation

After installation, test it:

```powershell
C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe test_installation.py
```

## Run the Application

Once installed, you can run:

```powershell
C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe quick_start.py
```

Or:

```powershell
C:/Users/rumma/AppData/Local/Programs/Python/Python311/python.exe asl_translator_app.py
```

## Common Issues

### If you get permission errors:
Add `--user` flag:
```powershell
$python -m pip install --user -r requirements.txt
```

### If installation is slow:
Use a different mirror:
```powershell
$python -m pip install -i https://pypi.org/simple/ -r requirements.txt
```

### If mediapipe installation fails:
This is common on Windows. Try:
```powershell
$python -m pip install mediapipe --no-deps
$python -m pip install opencv-python numpy protobuf
```

