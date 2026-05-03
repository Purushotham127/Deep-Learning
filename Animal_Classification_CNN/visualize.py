"""Visualize dataset and model predictions"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config import CLASSES, IMG_SIZE
from inference import AnimalClassifier


def visualize_dataset(num_images_per_class=3):
    """Visualize sample images from dataset"""
    fig, axes = plt.subplots(len(CLASSES), num_images_per_class, figsize=(12, 10))
    
    for row, class_name in enumerate(CLASSES):
        class_dir = os.path.join("dataset", class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for col in range(num_images_per_class):
            ax = axes[row, col] if len(CLASSES) > 1 else axes[col]
            
            if col < len(images):
                img_path = os.path.join(class_dir, images[col])
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(class_name, fontsize=12, fontweight='bold')
                ax.set_title(images[col][:20], fontsize=9)
            
            ax.axis('off')
    
    plt.suptitle('Animal Classification Dataset Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logs/dataset_samples.png', dpi=100, bbox_inches='tight')
    print("Dataset visualization saved to logs/dataset_samples.png")
    plt.close()


def visualize_predictions(num_images=5):
    """Visualize model predictions"""
    try:
        classifier = AnimalClassifier()
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return
    
    # Collect sample images
    sample_images = []
    for class_name in CLASSES:
        class_dir = os.path.join("dataset", class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images[:num_images]:
                sample_images.append(os.path.join(class_dir, img_file))
    
    # Predictions
    num_cols = min(3, len(sample_images))
    num_rows = (len(sample_images) + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten() if len(sample_images) > 1 else [axes]
    
    for idx, image_path in enumerate(sample_images):
        ax = axes[idx]
        
        # Load and display image
        img = Image.open(image_path).convert('RGB')
        ax.imshow(img)
        
        # Get prediction
        result = classifier.predict(image_path, top_k=3)
        
        # Display prediction info
        title = f"Predicted: {result['top_class']}\n({result['confidence']})"
        ax.set_title(title, fontsize=11, fontweight='bold', color='green')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions on Sample Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logs/predictions_visualization.png', dpi=100, bbox_inches='tight')
    print("Predictions visualization saved to logs/predictions_visualization.png")
    plt.close()


def main():
    """Main visualization function"""
    print("Generating visualizations...")
    
    os.makedirs('logs', exist_ok=True)
    
    print("1. Creating dataset samples visualization...")
    visualize_dataset()
    
    print("2. Creating model predictions visualization...")
    visualize_predictions()
    
    print("\nAll visualizations saved to logs/ directory")


if __name__ == "__main__":
    main()
