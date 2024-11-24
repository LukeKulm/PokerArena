import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np

def extract_moves(state):
  """gets the moves the player made in the game"""
  return 0
  


def aggregate(encodings):
    """
    Creates .pt files of information from extracted encodings of pluribus games
    """
    #get game from encoding
    #get moves from encoding
    state = encodings
    actions = extract_moves(state)
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
  hist_dir = "data/pluribus_extracted"
  files = [os.path.join(hist_dir, f) for f in os.listdir(hist_dir) if f.endswith('.txt')]
  for file in files:
    with open(file, "r") as file:
      lines = file.readlines()
      encodings = [line.strip() for line in lines]
      encodings = lines
      print(encodings)
      aggregate(encodings)