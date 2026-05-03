"""Configuration for Animal Classification CNN Model"""

# Dataset parameters
DATASET_DIR = "dataset"
CLASSES = ["Butterfly", "Dog", "Elephant", "Hen", "Horse"]
NUM_CLASSES = len(CLASSES)

# Image parameters
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMG_STD = [0.229, 0.224, 0.225]

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model parameters
MODEL_ARCHITECTURE = "custom_cnn"  # Options: "custom_cnn", "resnet18"
DROPOUT_RATE = 0.5

# Device
DEVICE = "cpu"  # Will be overridden if GPU is available

# Paths
MODEL_SAVE_PATH = "saved_models/animal_classifier.pth"
LOGS_DIR = "logs"
