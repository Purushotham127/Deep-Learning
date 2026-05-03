"""Quick setup and utility functions"""

import os
import sys
import subprocess
from config import DATASET_DIR, CLASSES


def check_dataset():
    """Check if dataset exists and is properly structured"""
    print("Checking dataset structure...")
    print(f"Dataset directory: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory '{DATASET_DIR}' not found!")
        return False
    
    missing_classes = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            missing_classes.append(class_name)
            print(f"❌ Missing class directory: {class_name}")
        else:
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            print(f"✓ {class_name}: {len(images)} images")
    
    if missing_classes:
        print(f"\n⚠️  Missing classes: {', '.join(missing_classes)}")
        return False
    
    print("\n✓ Dataset structure is valid!")
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    required = ['torch', 'torchvision', 'PIL', 'numpy', 'sklearn', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package if package != 'PIL' else 'PIL', fromlist=[''])
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True


def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    directories = ['saved_models', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory}/")
        else:
            print(f"✓ {directory}/ already exists")
    
    print("✓ Directory setup complete!")


def print_quick_start():
    """Print quick start instructions"""
    print("\n" + "=" * 60)
    print("QUICK START GUIDE")
    print("=" * 60)
    
    print("\n1️⃣  TRAINING THE MODEL:")
    print("   python train.py")
    print("   - This will train the CNN model on your dataset")
    print("   - Best model will be saved to saved_models/animal_classifier.pth")
    print("   - Training history will be saved to logs/training_history.json")
    
    print("\n2️⃣  EVALUATING THE MODEL:")
    print("   python evaluate.py")
    print("   - This will evaluate the trained model")
    print("   - Generates confusion matrix and accuracy reports")
    
    print("\n3️⃣  MAKING PREDICTIONS:")
    print("   python inference.py")
    print("   - Make predictions on sample images")
    print("   - Or use the AnimalClassifier class in your own code")
    
    print("\n4️⃣  VISUALIZING RESULTS:")
    print("   python visualize.py")
    print("   - Creates visualizations of dataset and predictions")
    
    print("\n📊 CONFIGURATION:")
    print("   Edit config.py to customize:")
    print("   - BATCH_SIZE: Batch size for training")
    print("   - LEARNING_RATE: Learning rate for optimizer")
    print("   - NUM_EPOCHS: Number of training epochs")
    print("   - IMG_SIZE: Input image size")
    
    print("\n📖 MORE INFO:")
    print("   See README.md for detailed documentation")
    print("=" * 60 + "\n")


def main():
    """Run setup checks and print info"""
    print("=" * 60)
    print("ANIMAL CLASSIFICATION CNN - SETUP CHECK")
    print("=" * 60)
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❓ Do you want to install dependencies? (yes/no)")
        response = input().strip().lower()
        if response in ['yes', 'y']:
            if not install_dependencies():
                sys.exit(1)
        else:
            print("Please install dependencies manually using: pip install -r requirements.txt")
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Print instructions
    if dataset_ok and deps_ok:
        print_quick_start()
        print("✅ Setup complete! You're ready to start training!\n")
    else:
        print("⚠️  Please fix the issues above before training.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
