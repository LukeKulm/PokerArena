import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

TRAIN_DATA_FILE_PATH = "./data/poker_hands/poker-hand-training-true.data"
TEST_DATA_FILE_PATH = "./data/poker_hands/poker-hand-testing.data"


class PokerHandPredictorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PokerHandAndStrengthDataset(Dataset):
    def __init__(self, data_path):

        self.features = []
        self.labels = []

        with open(data_path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.features.append([float(val) for val in row[:-1]])
                self.labels.append(float(row[-1]))

        self.features = torch.tensor(self.features, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def train(model, data_loader, lr=1e-3, epochs=1000):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss}")


train_dataset = PokerHandAndStrengthDataset(TRAIN_DATA_FILE_PATH)
train_data_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
model = PokerHandPredictorModel()

train(model, train_data_loader)


def test(model, data_loader):
    total_correct = 0
    total_predictions = 0

    for inputs, targets in data_loader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs, 1)
        total_predictions += targets.shape[0]
        total_correct += torch.sum((predicted == targets))

    print(f"Model Accuracy: {total_correct/total_predictions}")


test_dataset = PokerHandAndStrengthDataset(TEST_DATA_FILE_PATH)
test_data_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

test(model, test_data_loader)
