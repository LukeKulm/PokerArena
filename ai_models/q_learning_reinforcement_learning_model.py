import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np

StateActionNextStateInstance = namedtuple(
    'StateActionNextStateInstance', ('curr_state', 'curr_action', 'reward', 'next_state'))


class DataBuffer(object):
    def __init__(self, maxsize=500):
        self.buffer = deque(maxlen=maxsize)

    def add(self, state, action, reward, next_state):
        self.buffer.append(StateActionNextStateInstance(
            state, action, reward, next_state))

    def sample(self, batch_size=1000):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = [torch.from_numpy(state).float() for state in states]
        actions = [torch.tensor(action) for action in actions]
        rewards = [torch.tensor(reward).float() for reward in rewards]
        next_states = [torch.from_numpy(next_state).float()
                       for next_state in next_states]
        return (torch.stack(states),
                torch.stack(actions),
                torch.stack(rewards),
                torch.stack(next_states))


class PokerQNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size, hidden_sizes=[32, 64, 32], gamma=0.99):
        super().__init__()
        self.action_space_size = action_space_size
        self.fc1 = nn.Linear(state_space_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], action_space_size)
        self.gamma = gamma

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

    def get_action(self, state, epsilon):
        # takes as input numpy.ndarray
        # epsilon greedy to explore action space
        # might be other functions to try as well such as multiarmed bandit?
        state = torch.from_numpy(state).float()
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            q_function_values = self.forward(state)
            return int(torch.argmax(q_function_values))


def train_q_network(q_network: PokerQNetwork, buffer: DataBuffer, batch_size=100, learning_rate=1e-3):
    if len(buffer.buffer) >= batch_size:
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        curr_states, curr_actions, rewards, next_states = buffer.sample(
            batch_size)

        network_q_functions = q_network.forward(curr_states)
        network_q_values = network_q_functions[torch.arange(
            network_q_functions.size(0)), curr_actions]

        target_qs, _ = torch.max(q_network.forward(next_states), dim=1)
        target_q_values = q_network.gamma * target_qs + rewards
        loss = loss_fn(network_q_values, target_q_values)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
def supervised_finetune(q_network, expert_states, expert_actions, epochs=10, batch_size=100, learning_rate=1e-3):
    dataset = 5 #TODO: fix this
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss() #TODO, probably change this since want bet amount to be included as part of loss, but also could just focus on action and not bet amount
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for states, actions in dataloader:
            q_values = q_network.forward(states)
            
            loss = loss_fn(q_values, actions.long())
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    