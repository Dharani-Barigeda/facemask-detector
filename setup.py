#!/usr/bin/env python3
"""
FaceMask-Plus Setup Script
This script helps you set up the project and create the necessary directories.
"""

import os
import sys
import subprocess

def create_dataset_structure():
    """Create the dataset directory structure."""
    print("Creating dataset directory structure...")
    
    base_dirs = ['dataset/train', 'dataset/val']
    classes = ['mask', 'no_mask', 'incorrect_mask']
    
    for base_dir in base_dirs:
        for class_name in classes:
            dir_path = os.path.join(base_dir, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
    
    print("Dataset structure created successfully!")
    print("\nNext steps:")
    print("1. Add your training images to dataset/train/")
    print("2. Add your validation images to dataset/val/")
    print("3. Run: python train_mask_detector.py")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def main():
    print("FaceMask-Plus Setup")
    print("==================")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Create dataset structure
    create_dataset_structure()
    
    print("\nSetup completed successfully!")
    print("\nTo get started:")
    print("1. Add your images to the dataset directories")
    print("2. Train the model: python train_mask_detector.py")
    print("3. Run detection: python detect_mask_video.py")
    print("4. Launch web app: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()

