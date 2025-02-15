import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import datetime
import time
import numpy as np
import pandas as pd 

## TO save the full print
torch.set_printoptions(profile="full")

EPOCHS = 5
BATCH_SIZE = 64

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
    
def save_to_csv(model, rank, epoch):

    model_values = []
    for param in model.parameters(): 
        # For the network you have bias and weight 
        # the biases are 100 for the first layer since there are 100 neurons 
        # the weights are 100*784
        # same goes for the second layer
        model_values.append(param.detach().cpu().numpy().flatten())
        
    single_row = np.concatenate(([epoch], *model_values))
    num_params = single_row.shape[0]-1
    columns = ['epoch'] + [str(i+1) for i in range(num_params)]
    df = pd.DataFrame([single_row], columns=columns)
    csv_filename = f'traces/distributed_parameters_{rank}.csv'

    # Check if the CSV file exists
    if os.path.exists(csv_filename):
        # Append new row without writing the header
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        # Create a new CSV file with header
        df.to_csv(csv_filename, index=False)

def broadcast_params(model, src=0):
    """Broadcast all parameters from the parameter server to workers."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

def send_gradients(rank, model, dst=0):
    """Send gradients from worker to parameter server."""

    for i, param in enumerate(model.parameters()):
        grad_to_send = param.grad.data
        dist.send(grad_to_send, dst=dst)

def receive_gradients(model, src_ranks):
    """
    Parameter server receives gradients from each worker and
    accumulates them into param.grad.
    """
    with torch.no_grad():
        for param in model.parameters():
            grads = []
            for src in src_ranks:
                grad_recv = torch.zeros_like(param.data)
                dist.recv(grad_recv, src=src)
                grads.append(grad_recv)
            total_grad = torch.stack(grads).sum(dim=0)
            param.grad = total_grad / len(src_ranks)

def load_train_data(rank, world_size, batch_size=BATCH_SIZE):
    """
    Load and partition the MNIST training dataset among workers.
    Since rank 0 is the parameter server, we assign training data only
    to workers (ranks >=1). Here, the DistributedSampler is used to ensure
    each worker gets a unique subset of the data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Since rank 0 is the parameter server, set num_replicas to world_size - 1
    # and worker's local rank becomes (rank - 1)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size - 1,
        rank=rank - 1,
        shuffle=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_loader

def load_test_data():
    """
    Only the parameter server (rank=0) uses the test set for evaluation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    
    return accuracy

def train(rank, world_size, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # Create model
    model = SimpleNet()

    if rank == 0:
        # Parameter server: set device and load test data for evaluation
        test_loader = load_test_data()
        device = torch.device('cpu')  # or use 'cuda' if available
        model.to(device)
    else:
        # Workers: load a partitioned subset of the MNIST training set.
        train_loader = load_train_data(rank, world_size, batch_size)
        num_batches = len(train_loader)

    # Set up optimizer (using same hyperparameters for simplicity)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Broadcast initial parameters from PS (rank 0) to all workers
    broadcast_params(model, src=0)

    for epoch in range(epochs):
        if rank == 0:
            # ----------------- PARAMETER SERVER -----------------
            # For simplicity, assume each worker has the same number of batches.
            worker_ranks = list(range(1, world_size))
            # Typically, num_batches_for_this_demo should equal the number of batches
            # in each worker's loader. You can compute it dynamically, but here we assume:
            num_batches_for_this_demo = len(load_train_data(1, world_size, batch_size))
            for update_i in range(num_batches_for_this_demo):
                # Receive and aggregate gradients from all workers
                receive_gradients(model, worker_ranks)
                # Update the model
                optimizer.step()
                optimizer.zero_grad()
                # Broadcast updated parameters back to all workers
                broadcast_params(model, src=0)
            # Evaluate after completing all updates for the epoch
            acc = evaluate(model, test_loader, device=device)
            print(f"[Parameter Server] Epoch {epoch+1}/{epochs} done. Accuracy: {acc:.2f}%")
            save_to_csv(model, rank, epoch)
        else:
            # ----------------- WORKER -----------------
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                # Send computed gradients to the parameter server
                send_gradients(rank, model, dst=0)
                # Receive updated parameters from the parameter server
                broadcast_params(model, src=0)
            print(f"[Worker {rank}] Finished epoch {epoch+1}/{epochs}")
            save_to_csv(model, rank, epoch)

    dist.barrier()
    if rank == 0:
        print("[Parameter Server] Training complete.")
    else:
        print(f"[Worker {rank}] Done.")

if __name__ == '__main__':
    print("################")
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    print("World size:", world_size)
    print("Rank:", rank)
    print("################")

    # Initialize process group
    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)
    )

    if rank == 0:
        start = time.time()

    if world_size == 1:
        print("Warning: WORLD_SIZE=1 => No workers. Increase WORLD_SIZE to at least 2.")

    train(rank, world_size, epochs=EPOCHS, batch_size=BATCH_SIZE)

    dist.destroy_process_group()

    if rank == 0:
        end = time.time()
        print(f"Training time: {round(end - start, 2)}s")
