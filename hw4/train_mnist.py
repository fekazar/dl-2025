import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lion import Lion
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return total_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def plot_metrics(lion_metrics, adam_metrics, save_path='comparison.png'):
    epochs = range(1, len(lion_metrics['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, lion_metrics['train_loss'], 'b-', label='Lion')
    plt.plot(epochs, adam_metrics['train_loss'], 'r-', label='Adam')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lion_metrics['test_acc'], 'b-', label='Lion')
    plt.plot(epochs, adam_metrics['test_acc'], 'r-', label='Adam')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_with_optimizer(optimizer_name, model, device, train_loader, test_loader, epochs):
    if optimizer_name == 'Lion':
        optimizer = Lion(model.parameters(), lr=1e-4)
    else:  # Adam
        optimizer = Adam(model.parameters(), lr=1e-4)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    
    return {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'test_acc': test_accuracies
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    epochs = 10
    
    # Train with Lion
    print("\nTraining with Lion optimizer...")
    lion_model = SimpleCNN().to(device)
    lion_metrics = train_with_optimizer('Lion', lion_model, device, train_loader, test_loader, epochs)
    
    # Train with Adam
    print("\nTraining with Adam optimizer...")
    adam_model = SimpleCNN().to(device)
    adam_metrics = train_with_optimizer('Adam', adam_model, device, train_loader, test_loader, epochs)
    
    # Plot and save comparison
    plot_metrics(lion_metrics, adam_metrics)
    print("\nComparison plot saved as 'comparison.png'")
    
    # Print final metrics
    print("\nFinal Results:")
    print(f"Lion - Best accuracy: {max(lion_metrics['test_acc']):.2f}%")
    print(f"Adam - Best accuracy: {max(adam_metrics['test_acc']):.2f}%")

if __name__ == '__main__':
    main() 