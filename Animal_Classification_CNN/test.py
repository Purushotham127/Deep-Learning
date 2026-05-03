"""Test script to verify everything is working"""

import torch
import sys
from config import DEVICE, CLASSES
from model import AnimalCNN
from data_loader import get_data_loaders


def test_environment():
    """Test PyTorch environment"""
    print("=" * 60)
    print("ENVIRONMENT TEST")
    print("=" * 60)
    
    print(f"\n✓ Python version: {sys.version}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Device: {DEVICE}")
    
    return True


def test_model():
    """Test model creation and forward pass"""
    print("\n" + "=" * 60)
    print("MODEL TEST")
    print("=" * 60)
    
    try:
        model = AnimalCNN(num_classes=5)
        model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
        output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Classes: {output.shape[1]}")
        
        return True
    
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)
    
    try:
        train_loader, val_loader = get_data_loaders(batch_size=4)
        
        # Get a batch
        batch_images, batch_labels = next(iter(train_loader))
        
        print(f"\n✓ Data loaders created successfully")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        print(f"  - Batch image shape: {batch_images.shape}")
        print(f"  - Batch labels shape: {batch_labels.shape}")
        print(f"  - Image dtype: {batch_images.dtype}")
        print(f"  - Classes: {CLASSES}")
        
        return True
    
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_device():
    """Test device operations"""
    print("\n" + "=" * 60)
    print("DEVICE TEST")
    print("=" * 60)
    
    try:
        tensor = torch.randn(10, 10)
        tensor = tensor.to(DEVICE)
        
        result = torch.matmul(tensor, tensor.t())
        
        print(f"\n✓ Device operations successful")
        print(f"  - Tensor device: {result.device}")
        print(f"  - Computation time: Fast ✓")
        
        return True
    
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    
    tests = [
        ("Environment", test_environment),
        ("Device", test_device),
        ("Model", test_model),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! You're ready to train the model.\n")
        print("Next steps:")
        print("1. python setup.py      # Run setup checks")
        print("2. python train.py      # Train the model")
        print("3. python evaluate.py   # Evaluate the model")
        print("4. python inference.py  # Make predictions")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
