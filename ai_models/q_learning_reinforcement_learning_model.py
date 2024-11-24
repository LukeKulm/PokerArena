import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np

# NOTE
# Code inspired by a CS 4789 project and also:
# - Medium article on Deep Q-Learning with PyTorch (https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc)
# - PyTorch tutorial on Q-Learning (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
# Adapted according to custom poker environment and requirements.
class QLearningException(Exception):
    pass

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx]).float(), torch.tensor(self.actions[idx])

class DataBuffer(object):
    def __init__(self, maxlen=500):
        self.buffer = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state):
        # may want to add done state to indicate that either this
        # player has won or that this player has a balance of 0
        self.buffer.append((state, action, reward, next_state))

    def weighted_sample(self, batch_size=50):
        # TODO: add some constant here to tune the weighting
        buffer_length = len(self.buffer)

        if buffer_length < batch_size:
            raise QLearningException("buffer length less than batch_size")

        # prioritize items that were more recent
        weights = np.linspace(1, 2, buffer_length)
        selection_probabilities = weights / np.sum(weights)
        sampled_indexes = np.random.choice(buffer_length, size = batch_size, p = selection_probabilities)
        batch = []
        for i in sampled_indexes:
            batch.append(self.buffer[i])
            
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
    def __init__(self, state_space_size, action_space_size, hidden_sizes=[64, 128, 64], gamma=0.98, lr_negative_slope = 1e-2, softmax = False):
        super().__init__()
        self.action_space_size = action_space_size
        self.fc1 = nn.Linear(state_space_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], action_space_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=lr_negative_slope)
        self.gamma = gamma
        self.softmax = softmax

    def forward(self, state):
        x = self.fc1(state)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        return x
    
    def epsilon_greedy_action_selection(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            state = torch.from_numpy(state).float()
            with torch.no_grad():
                q_function_values = self.forward(state)
            return int(torch.argmax(q_function_values))
        
    def soft_max_action_selection(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            q_values = self.forward(state)
            action_probabilities = torch.softmax(q_values, dim = -1)
            action = torch.multinomial(action_probabilities, 1).item()
            return action

    def select_action(self, state, epsilon=1e-2):
        # takes as input numpy.ndarray
        if self.softmax:
            # softmax approach
            return self.soft_max_action_selection(state)
        else:
            # epsilon greedy
            return self.epsilon_greedy_action_selection(state, epsilon)



def train_q_network(q_network: PokerQNetwork, buffer: DataBuffer, batch_size=100, learning_rate=1e-3):
    if len(buffer.buffer) >= batch_size:
        # we may also want to consider huber loss or SmoothL1Loss or etc...
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        curr_states, curr_actions, rewards, next_states = buffer.weighted_sample(batch_size)

        network_q_functions = q_network(curr_states)
        network_q_values = network_q_functions[torch.arange(
            network_q_functions.size(0)), curr_actions]

        next_state_q_values, _ = torch.max(q_network(next_states), dim=1)
        r_plus_gamma_times_next_qs = q_network.gamma * next_state_q_values + rewards
        loss = loss_fn(network_q_values, r_plus_gamma_times_next_qs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
def supervised_finetune(q_network, expert_data, epochs=10, batch_size=100, learning_rate=1e-3):
    #TODO: need to extract actions and need to turn actions into 1 of 14 actions
    dataset = ExpertDataset(expert_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss() #TODO, probably change this since want bet amount to be included as part of loss, but also could just focus on action and not bet amount
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for states, actions in dataloader:
            q_values = q_network.forward(states)
            
            #whats this actions.long() buisness
            loss = loss_fn(q_values, actions.long())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    