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

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load the two datasets and image loaders
train_dataset = datasets.ImageFolder(root='D:\\Dropbox\\UMA Augusta\\PhD\\Research Thesis\\brain_tumor_mri_dataset\\Training', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model, loss function, optimizer
model = models.resnet50(weights=ResNet50_Weights.DEFAULT) 
model.to(device)  # Move model to the appropriate device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start the training loop
num_epochs = 10  
for epoch in range(num_epochs):
    model.train()  
    training_loss = 0.0  
    # Iterate over batches of training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)#Move data to device if cude available or not
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        training_loss += loss.item()

    # Print the average loss for this epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {training_loss / len(train_loader):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'mri_model.pth')
print("Model saved as 'mri_model.pth'") 