# Animal Classification CNN with PyTorch

A deep learning project for classifying animal images (Butterfly, Dog, Elephant, Hen, Horse) using Convolutional Neural Networks (CNN) built with PyTorch. Optimized for CPU training.

## Project Structure

```
Animal_Classification_CNN/
├── dataset/                    # Animal image dataset
│   ├── Butterfly/
│   ├── Dog/
│   ├── Elephant/
│   ├── Hen/
│   └── Horse/
├── saved_models/              # Trained model checkpoints
├── logs/                       # Training logs and visualizations
├── config.py                   # Configuration parameters
├── model.py                    # CNN model architectures
├── data_loader.py             # Data loading and preprocessing
├── train.py                   # Training script
├── evaluate.py                # Model evaluation
├── inference.py               # Single/batch inference
├── visualize.py               # Dataset and prediction visualization
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Two CNN Architectures**: 
  - `AnimalCNN`: Full-featured CNN with 4 convolutional blocks (256+ filters)
  - `SimpleAnimalCNN`: Lightweight CNN for faster CPU training
  
- **Data Augmentation**: Random horizontal flips, rotation, color jitter, and affine transforms
- **Training Features**:
  - Early stopping with patience
  - Learning rate scheduling
  - Checkpoint saving for best model
  - Detailed training history logging
  
- **Evaluation**: 
  - Per-class accuracy
  - Confusion matrix visualization
  - Classification report
  - Training history plots
  
- **Inference**: 
  - Single image prediction
  - Batch prediction
  - Top-K predictions with confidence scores

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Using Python venv
python -m venv venv
.\venv\Scripts\activate  # On Windows

# OR using Conda
conda create -n animal_cnn python=3.9
conda activate animal_cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

```bash
python train.py
```

The script will:
- Load and display dataset statistics
- Create and display model architecture
- Train for up to 30 epochs (with early stopping)
- Save the best model to `saved_models/animal_classifier.pth`
- Save training history to `logs/training_history.json`

**Training Time**: ~5-15 minutes per epoch on CPU (depends on dataset size)

**Key Parameters** (modify in `config.py`):
- `NUM_EPOCHS`: Number of training epochs (default: 30)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `VALIDATION_SPLIT`: Fraction of data for validation (default: 0.2)

### 2. Evaluating the Model

```bash
python evaluate.py
```

This will:
- Load the trained model
- Compute accuracy and classification metrics
- Generate confusion matrix visualization
- Create per-class accuracy bar chart
- Print detailed classification report

### 3. Making Predictions

#### Single Image Prediction

```python
from inference import AnimalClassifier

classifier = AnimalClassifier()
result = classifier.predict("path/to/image.jpg", top_k=3)

print(f"Top prediction: {result['top_class']}")
print(f"Confidence: {result['confidence']}")
```

#### Batch Prediction

```bash
python inference.py
```

Or programmatically:

```python
from inference import AnimalClassifier

classifier = AnimalClassifier()
results = classifier.predict_batch([
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "path/to/image3.jpg"
])

for res in results:
    print(f"{res['image']}: {res['result']['top_class']}")
```

### 4. Visualizing Results

```bash
python visualize.py
```

This generates:
- `logs/dataset_samples.png`: Sample images from each class
- `logs/predictions_visualization.png`: Model predictions on sample images

## Configuration

Edit `config.py` to customize:

```python
# Model
NUM_CLASSES = 5
DROPOUT_RATE = 0.5

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2

# Image
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMG_STD = [0.229, 0.224, 0.225]

# Device
DEVICE = "cpu"  # Auto-detected, uses GPU if available
```

## Model Architecture

### AnimalCNN

4 convolutional blocks with batch normalization:
- Block 1: 32 filters, 3x3 kernels
- Block 2: 64 filters, 3x3 kernels
- Block 3: 128 filters, 3x3 kernels
- Block 4: 256 filters, 3x3 kernels

Followed by:
- Global average pooling
- 2 fully connected layers (512, 256 units)
- Dropout for regularization
- Output layer with 5 classes

**Total Parameters**: ~2.2M trainable parameters

## Data Augmentation

Training images are augmented with:
- Random horizontal flips (50% probability)
- Random rotation (±15 degrees)
- Random color jitter (brightness, contrast, saturation ±0.2)
- Random affine transformation (±10% translation)

Validation images are only resized and normalized (no augmentation).

## Tips for Better Performance

1. **Dataset**: Ensure each class has at least 50-100 images
2. **Image Quality**: Use high-quality images with good lighting
3. **Augmentation**: Increase augmentation if overfitting occurs
4. **Batch Size**: Increase batch size for more stable gradients (if memory allows)
5. **Learning Rate**: Decrease if loss oscillates, increase if training plateaus
6. **Epochs**: Increase for better convergence (early stopping prevents overfitting)

## Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` in `config.py`

**Poor Validation Accuracy**: 
- Increase training data
- Increase `NUM_EPOCHS`
- Decrease `LEARNING_RATE`
- Increase data augmentation

**Training Too Slow**: Use `SimpleAnimalCNN` instead of `AnimalCNN` in `train.py`

**Model Not Found Error**: Train the model first using `python train.py`

## System Requirements

- **Python**: 3.8+
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: ~1GB for dataset + model
- **GPU**: Optional (model runs on CPU)

## Performance Notes

On CPU:
- Training ~10-20 seconds per batch (32 images)
- ~5-15 minutes per epoch depending on dataset size
- Inference ~1-2 seconds per image

## License

This project is provided as-is for educational purposes.

## Author

Created for animal image classification using PyTorch CNN.
