
#!/usr/bin/env python3
"""
Installation Test Script
=======================

This script tests if all dependencies are properly installed.
"""

import sys

def test_dependencies():
    """Test if all required dependencies are available"""

    print("Testing NSFW Scanner Dependencies")
    print("=" * 40)
    print()

    # Test basic Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ is required")
        return False
    else:
        print("✅ Python version OK")
    print()

    # Test core dependencies
    dependencies = [
        ("torch", "PyTorch for AI models"),
        ("torchvision", "PyTorch vision utilities"),
        ("transformers", "Hugging Face transformers"),
        ("timm", "PyTorch image models"),
        ("PIL", "Pillow image processing"),
        ("cv2", "OpenCV image processing"),
        ("numpy", "Numerical computing"),
        ("tkinter", "GUI framework")
    ]

    missing = []

    for module, description in dependencies:
        try:
            if module == "PIL":
                import PIL
            elif module == "cv2":
                import cv2
            else:
                __import__(module)
            print(f"✅ {module:15} - {description}")
        except ImportError:
            print(f"❌ {module:15} - {description} (MISSING)")
            missing.append(module)

    print()

    # Test optional dependencies
    print("Optional Dependencies:")
    print("-" * 20)

    optional_deps = [
        ("opennsfw2", "OpenNSFW2 model"),
        ("nudenet", "Detailed body part detection")
    ]

    for module, description in optional_deps:
        try:
            __import__(module)
            print(f"✅ {module:15} - {description}")
        except ImportError:
            print(f"⚠️  {module:15} - {description} (optional, but recommended)")

    print()

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"🚀 GPU acceleration available: {gpu_name} ({gpu_memory}GB)")
        else:
            print("💻 GPU acceleration not available (will use CPU)")
    except:
        print("❓ Unable to check GPU status")

    print()

    if missing:
        print("❌ Installation incomplete!")
        print("Missing dependencies:", ", ".join(missing))
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All core dependencies installed successfully!")
        print("You can now run: python nsfw_image_scanner.py")
        return True

if __name__ == "__main__":
    success = test_dependencies()
    sys.exit(0 if success else 1)
