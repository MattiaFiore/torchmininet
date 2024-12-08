import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time 

#torch.set_num_threads(4)  # Use 4 CPU threads for the training process

# 1. Set up transforms to normalize the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the data (grayscale)
])

# 2. Load Fashion-MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Define a deeper Convolutional Neural Network (CNN) for Fashion-MNIST
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer to reduce dimensionality
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Flattened from 128x3x3
        self.fc2 = nn.Linear(512, 10)  # Output layer with 10 classes
        
        # Batch normalization (to stabilize training)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout layer (to avoid overfitting)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 + ReLU + BatchNorm + Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 + ReLU + BatchNorm + Pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 + ReLU + BatchNorm + Pooling
        x = x.view(-1, 128 * 3 * 3)  # Flatten for the fully connected layer
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = self.dropout(x)  # Dropout for regularization
        x = self.fc2(x)  # Output layer (logits)
        return x

# 4. Set device for training (CPU in this case)
device = torch.device("cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the device
model = DeepCNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
start = time.time()
epochs = 10  # You can increase the epochs for a deeper model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print the average loss and accuracy for this epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = (correct / total) * 100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 6. Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = (correct / total) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

end = time.time()
print(f"Tempo impiegato: {round(end-start,6)}s")