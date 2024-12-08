import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision import datasets, transforms
import datetime
import time 


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

def broadcast_params(model, src=0):
    # Broadcast all parameters from the source (parameter server) to workers
    for param in model.parameters():
        dist.broadcast(param.data, src)

def send_gradients(model, dst=0):
    # Send gradients from worker to parameter server
    for param in model.parameters():
        dist.send(param.grad.data, dst=dst)

def receive_gradients(model, src_ranks):
    # Parameter server receives gradients from each worker and accumulates them
    with torch.no_grad():
        for param in model.parameters():
            grads = []
            for src in src_ranks:
                grad_recv = torch.zeros_like(param.data)
                dist.recv(grad_recv, src=src)
                grads.append(grad_recv)
            total_grad = torch.stack(grads).sum(dim=0)
            param.grad = total_grad

def load_data(rank, world_size):
    # Download MNIST dataset (train)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # If rank 0 is the parameter server, it does not need training data.
    if rank == 0:
        return None

    total_size = len(train_dataset)
    worker_indices = list(range((rank-1), total_size, (world_size-1)))
    subset = Subset(train_dataset, worker_indices)

    if len(worker_indices) == 0:
        print(f"Warning: Worker {rank} has no samples. Check dataset size and world_size.")
    
    loader = DataLoader(subset, batch_size=64, shuffle=True)
    return loader

def load_test_data():
    # Only the parameter server (rank=0) will use the test set for evaluation
    transform = transforms.Compose([transforms.ToTensor()])
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

def train(rank, world_size, epochs=2):
    # Create model
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Load data
    train_loader = load_data(rank, world_size)

    # If parameter server, load test data to evaluate after each epoch
    if rank == 0:
        test_loader = load_test_data()
    else:
        test_loader = None

    # Initialize parameters on workers
    if rank == 0:
        # Parameter Server: Initialize parameters and broadcast
        broadcast_params(model, src=0)
    else:
        # Workers: Receive initial parameters
        broadcast_params(model, src=0)

    # Start training
    for epoch in range(epochs):
        if rank == 0:
            # Parameter server loop
            num_updates = 100  # expecting 100 gradient exchanges per epoch
            for update_i in range(num_updates):
                # Receive gradients from all workers
                worker_ranks = list(range(1, world_size))
                receive_gradients(model, src_ranks=worker_ranks)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Broadcast updated parameters back to workers
                broadcast_params(model, src=0)

            # After finishing all updates for this epoch on the parameter server:
            # Evaluate and print accuracy
            accuracy = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs} completed on Parameter Server. Accuracy: {accuracy:.2f}%")
        else:
            # Worker training
            if train_loader is None:
                # If for some reason worker got no data, skip training
                print(f"Worker {rank} has no data to train on. Skipping...")
                # Still broadcast params after the epoch, so workers remain in sync
                continue

            iter_loader = iter(train_loader)
            for _ in range(100):  # match the number of updates the PS expects
                try:
                    data, target = next(iter_loader)
                except StopIteration:
                    # If out of data, just break
                    break
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # Send gradients to parameter server
                send_gradients(model, dst=0)

                # Receive updated parameters
                broadcast_params(model, src=0)

            # After finishing all updates for this epoch on the worker:
            print(f"Worker {rank} completed epoch {epoch+1}/{epochs}.")

    # Synchronize processes after training
    dist.barrier()

    if rank == 0:
        print("Training complete on parameter server.")
    else:
        print(f"Worker {rank} complete.")

if __name__ == '__main__':
    print("################")
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    print("World size:", world_size)
    print("Rank:", rank)
    print("################")
   
    # INITIALIZING THE GROUP PROCESS
    dist.init_process_group(
        backend='gloo', 
        init_method='env://', 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)
    )

    print("Riuscito ad inizializzare")
    if rank == 0: 
        start = time.time()

    # Make sure world_size > 1. If not, you'll have no workers and this won't really work as intended.
    if world_size == 1:
        print("Warning: WORLD_SIZE=1 means no workers. Increase world_size to at least 2 (1 PS + 1 worker).")

    epochs = 20
    train(rank, world_size, epochs=epochs)
    dist.destroy_process_group()
    if rank == 0: 
        end = time.time()
        print(f"Tempo occupato per il training: {round(end-start,4)}s")
