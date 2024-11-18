import torch
import torch.nn as nn
from collections import deque, namedtuple
import random
import numpy as np


class PokerQNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size, hidden_sizes=[32, 64, 32]):
        self.action_space_size = action_space_size
        self.fc1 = nn.Linear(state_space_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], action_space_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

    def get_action(self, state, epsilon):
        # epsilon greedy to explore action space
        # might be other functions to try as well such as multiarmed bandit?
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            q_function_values = self.forward(state)
            return int(torch.argmax(q_function_values))


StateActionNextStateInstance = namedtuple(
    'StateActionNextStateInstance', ('curr_state', 'curr_action', 'reward', 'next_state', 'done'))


class DataBuffer(object):
    def __init__(self, maxsize=100000):
        self.buffer = deque(maxlen=maxsize)

    def add(self, state, action, reward, next_state, done):
        # right now done doesn't really have a meaning I think?
        self.buffer.append(StateActionNextStateInstance(
            state, action, reward, next_state, done))

    def sample(self, batch_len):
        batch = random.sample(self.buffer, batch_len)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.stack(actions),
                torch.stack(rewards),
                torch.stack(next_states),
                torch.stack(dones).float())
