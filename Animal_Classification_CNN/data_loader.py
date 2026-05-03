"""Data Loading and Preprocessing for Animal Classification"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from config import (
    DATASET_DIR, IMG_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED, CLASSES
)


class AnimalDataset(Dataset):
    """Custom Dataset for loading animal images"""
    
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Directory with all the animal images organized by class
            transform (callable, optional): Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        # Load all image paths and labels
        for class_name in CLASSES:
            class_dir = os.path.join(img_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Check if file is an image
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Get data transforms for training and validation
    
    Args:
        train (bool): Whether to return training or validation transforms
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
    
    return transform


def get_data_loaders(batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """
    Create training and validation data loaders
    
    Args:
        batch_size (int): Batch size for data loaders
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset with training transforms
    full_dataset = AnimalDataset(
        DATASET_DIR,
        transform=get_transforms(train=True)
    )
    
    print(f"Total images found: {len(full_dataset)}")
    print(f"Class distribution:")
    for class_name, class_idx in full_dataset.class_to_idx.items():
        count = sum(1 for label in full_dataset.labels if label == class_idx)
        print(f"  {class_name}: {count} images")
    
    # Split into training and validation
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    # Create new validation dataset with validation transforms
    val_dataset.dataset = AnimalDataset(
        DATASET_DIR,
        transform=get_transforms(train=False)
    )
    val_indices = val_dataset.indices
    val_dataset.indices = val_indices
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for CPU training on Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def get_class_distribution(dataset):
    """Calculate class distribution in dataset"""
    class_counts = np.zeros(len(CLASSES))
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts
