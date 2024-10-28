import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#neural network layers and stuff
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x

#define ensemble class
class Ensemble:
    def __init__(self, num_models):
        self.num_models = num_models
        self.models = [ComplexNet() for _ in range(num_models)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]

    def train(self, train_loader):
        for model in self.models:
            model.train()
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            for model in self.models:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return total_loss / (len(train_loader) * self.num_models)

    def evaluate(self, test_loader):
        for model in self.models:
            model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                ensemble_outputs = torch.zeros(inputs.size(0), 10)
                for model in self.models:
                    outputs = model(inputs)
                    ensemble_outputs += outputs
                ensemble_outputs /= self.num_models
                loss = criterion(ensemble_outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(ensemble_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        return val_loss / len(test_loader), val_accuracy

#load MNIST dataset with data augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Instantiate the ensemble
ensemble = Ensemble(num_models=5)

#define loss function
criterion = nn.CrossEntropyLoss()

#training loop 
num_epochs = 10
for epoch in range(num_epochs):
    avg_train_loss = ensemble.train(train_loader)
    avg_val_loss, val_accuracy = ensemble.evaluate(test_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Avg Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy * 100:.2f}%')

#testing the ensemble
test_loss, test_accuracy = ensemble.evaluate(test_loader)
print(f'Final Test Accuracy: {test_accuracy * 100:.2f}%')
