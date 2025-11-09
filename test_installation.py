"""
Test script to verify all dependencies are installed correctly
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            __import__(module_name)
            module = sys.modules[module_name]
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        else:
            __import__(module_name)
            print(f"✅ {module_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: NOT INSTALLED")
        print(f"   Error: {e}")
        return False

def main():
    print("=" * 60)
    print("ASL Translator - Installation Test")
    print("=" * 60)
    print()
    
    print("Testing Python version...")
    print(f"Python: {sys.version}")
    print()
    
    print("Testing required packages...")
    print("-" * 60)
    
    results = []
    
    # Test required packages
    results.append(test_import("cv2", "OpenCV (opencv-python)"))
    results.append(test_import("mediapipe", "MediaPipe"))
    results.append(test_import("numpy", "NumPy"))
    results.append(test_import("scipy", "SciPy"))
    results.append(test_import("requests", "Requests"))
    results.append(test_import("streamlit", "Streamlit"))
    results.append(test_import("PIL", "Pillow"))
    
    print()
    print("-" * 60)
    
    if all(results):
        print("✅ All packages are installed correctly!")
        print()
        print("You can now run:")
        print("  python asl_translator_app.py")
        print("  or")
        print("  streamlit run streamlit_app.py")
        return 0
    else:
        print("❌ Some packages are missing!")
        print()
        print("To install missing packages, run:")
        print("  python -m pip install -r requirements.txt")
        print()
        print("Or use the installation script:")
        print("  install_windows.bat  (Windows)")
        print("  install_windows.ps1  (Windows PowerShell)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)

