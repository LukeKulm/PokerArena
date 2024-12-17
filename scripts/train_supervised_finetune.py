import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from ai_models.q_learning_reinforcement_learning_model import *


input_dim = 23  # state
num_actions = 14  # num actions
q_network = PokerQNetwork(input_dim, num_actions)

pretrained_weights_path = "saved_models/q_network.pth"

if torch.cuda.is_available():
    q_network.load_state_dict(torch.load(pretrained_weights_path))
#TODO try to do optimized on M1
else:
    q_network.load_state_dict(torch.load(pretrained_weights_path, map_location=torch.device("cpu")))

print("Pre-trained weights successfully loaded!")

# finetune
expert_data_path = "data/pluribus_policy.pt"
supervised_finetune(q_network, expert_data_path, epochs=10, batch_size=100, learning_rate=1e-3)
