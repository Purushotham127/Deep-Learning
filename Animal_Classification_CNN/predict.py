from inference import AnimalClassifier

classifier = AnimalClassifier()

# Predict on an external image
result = classifier.predict("C:\\Users\\Purushotham\\Desktop\\Animal_Classification_CNN\\predict_images\\dog_1.jpg", top_k=3)

print(f"Animal detected: {result['top_class']}")
print(f"Confidence: {result['confidence']}")
print(f"{result}")