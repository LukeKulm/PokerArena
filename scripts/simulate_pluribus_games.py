import sys
import argparse
import game
import torch
import os
import numpy as np
from player import PLAYER_TYPES, PLAYER_TYPES_THAT_REQUIRE_TORCH_MODELS
from scripts.utils import get_not_busted

def extract_moves(state):
  """gets the moves the player made in the game"""


def aggregate():
    """
    Creates .pt files of information from extracted encodings of pluribus games
    """
    #get game from encoding
    #get moves from encoding
    x1 = state
    y1 = moves
    new_data = torch.tensor(np.array(x1), dtype=torch.float32)
    new_labels = torch.tensor(np.array(y1), dtype=torch.float32)
    if not os.path.exists('data/pluribus_policy.pt'):
        torch.save((new_data, new_labels), 'data/pluribus_policy.pt')
    else:
        existing_data, existing_labels = torch.load(
            'data/pluribus_policy.pt', weights_only=True)
        updated_data = torch.cat((existing_data, new_data), dim=0)
        updated_labels = torch.cat((existing_labels, new_labels), dim=0)
        torch.save((updated_data, updated_labels), 'data/pluribus_policy.pt')


if __name__ == "__main__":
    aggregate()