"""Inference Script for Animal Classification CNN"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from config import CLASSES, IMG_SIZE, IMG_MEAN, IMG_STD, MODEL_SAVE_PATH
from model import AnimalCNN
import os


class AnimalClassifier:
    """Class for performing inference with trained model"""
    
    def __init__(self, model_path=MODEL_SAVE_PATH, device=None):
        """
        Initialize the classifier
        
        Args:
            model_path (str): Path to saved model checkpoint
            device (torch.device): Device to run model on
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = AnimalCNN(num_classes=5)
        
        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0
        
        # Normalize
        image_array = (image_array - np.array(IMG_MEAN)) / np.array(IMG_STD)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict(self, image_path, top_k=3):
        """
        Predict class for given image
        
        Args:
            image_path (str): Path to image file
            top_k (int): Number of top predictions to return
        
        Returns:
            dict: Predictions with probabilities
        """
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_prob[0], top_indices[0]):
            results.append({
                'class': CLASSES[idx.item()],
                'probability': prob.item(),
                'confidence': f"{prob.item()*100:.2f}%"
            })
        
        return {
            'predictions': results,
            'top_class': results[0]['class'],
            'confidence': results[0]['confidence']
        }
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image file paths
        
        Returns:
            list: List of predictions
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append({
                    'image': image_path,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'image': image_path,
                    'error': str(e)
                })
        
        return results


def main():
    """Example inference"""
    print("=" * 50)
    print("Animal Classification Inference")
    print("=" * 50)
    
    # Initialize classifier
    classifier = AnimalClassifier()
    
    # Example: Predict on a single image
    print("\nExample 1: Single image prediction")
    print("-" * 50)
    
    # Find a sample image
    sample_images = []
    for class_name in CLASSES:
        class_dir = os.path.join("dataset", class_name)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join(class_dir, img_file))
                    break
    
    if sample_images:
        image_path = sample_images[0]
        print(f"Image: {image_path}")
        
        result = classifier.predict(image_path, top_k=3)
        print(f"\nTop Prediction: {result['top_class']} ({result['confidence']})")
        print("\nAll Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['class']}: {pred['confidence']}")
    else:
        print("No sample images found in dataset")
    
    # Example: Batch prediction
    if len(sample_images) > 1:
        print("\n\nExample 2: Batch prediction")
        print("-" * 50)
        batch_results = classifier.predict_batch(sample_images[:3])
        
        for res in batch_results:
            if 'result' in res:
                print(f"\n{res['image']}")
                print(f"  Predicted: {res['result']['top_class']} ({res['result']['confidence']})")
            else:
                print(f"{res['image']}: Error - {res['error']}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
