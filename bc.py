import torch
import torch.nn as nn
import torch.optim as optim

class NN(nn.Module):
    def __init__(self, input_size=23, hidden_size=50, output_size=3):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_data, train_targets, num_epochs=500, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            outputs = self.forward(train_data)
            loss = criterion(outputs, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
        print("Training complete!")
    def evaluate(self, test_data, test_targets):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(test_data)
            criterion = nn.MSELoss()
            loss = criterion(predictions, test_targets)
            print(f'Evaluation Loss: {loss.item():.4f}')
        return loss.item()
    def save_checkpoint(model, file_path='bc_checkpoint.pth'):
        torch.save(model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

