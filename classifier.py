import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchmetrics import F1Score
import numpy as np
import torch.nn.functional as F 

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset and dataloader
test_dataset = datasets.ImageFolder(root='Kaggle/Augmented_Testing', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT) 
model.load_state_dict(torch.load('mri_model.pth', map_location=device, weights_only=True))
model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Evaluate mode
model.eval()

#accuracy function using the predicted output with the truth label
def calculate_accuracy(true_labels, predicted_labels):
    return np.sum(np.array(true_labels) == np.array(predicted_labels)) / len(true_labels)


results = []

#set epoch num for testing
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    truth_labels = []
    predicted_labels = []
    confidence_scores = []

    # validation/testing loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  
            loss = criterion(outputs, labels)
            # Apply softmax to get confidence scores
            probabilities = F.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]  # Get max confidence per sample
            
            # Get predicted labels
            _, predicted = torch.max(outputs, 1)
            # Update metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            truth_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            confidence_scores.extend(confidence.cpu().numpy())

    #calculate epoch results
    avg_loss = epoch_loss / len(test_loader)
    accuracy = correct / total
    avg_confidence = np.mean(confidence_scores)

    # Save results
    results.append({
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence
    })

# Print results 
print("\nFinal Testing Results Over 10 Epochs:\n")
for res in results:
    print(f"Epoch {res['epoch']} - Loss: {res['loss']:.4f}, Accuracy: {res['accuracy']:.2%}, Avg Confidence: {res['avg_confidence']:.4f}")
