import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time 
import numpy as np 
import pandas as pd 
import os 

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

# 
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Set device for training (CPU in this case)
device = torch.device("cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the device
model = SimpleNet().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
start = time.time()
epochs = 2  # You can increase the epochs for a deeper model
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

    model_values = []
    for param in model.parameters(): 
        # For the network you have bias and weight 
        # the biases are 100 for the first layer since there are 100 neurons 
        # the weights are 100*784
        # same goes for the second layer
        model_values.append(param.detach().cpu().numpy().flatten())
        
    single_row = np.concatenate(model_values)
    print(single_row)
    df = pd.DataFrame([single_row])
    csv_filename = 'traces/non_distributed_parameters.csv'

    # Check if the CSV file exists
    if os.path.exists(csv_filename):
        # Append new row without writing the header
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        # Create a new CSV file with header
        df.to_csv(csv_filename, index=False)


        

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