import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np

# 14 actions total
# action 0 is fold
# action 1 is call
# action 2 is raise by minimum amount
# action 3 is raise by double minimum amount of remaining_stack
# action 4 is raise by double minimum amount + 5% of remaining_stack
# action 5 is raise by double minimum amount + 10% of remaining_stack
# action 6 is raise by double minimum amount + 15% of remaining_stack
# action 7 is raise by double minimum amount + 20% of remaining_stack
# action 8 is raise by double minimum amount + 30% of remaining_stack
# action 9 is raise by double minimum amount + 40% of remaining_stack
# action 10 is raise by double minimum amount + 50% of remaining_stack
# action 11 is raise by double minimum amount + 65% of remaining_stack
# action 12 is raise by double minimum amount + 80% of remaining_stack
# action 13 is all in

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
    action = extract_moves(state)
    new_data = torch.tensor(np.array(state), dtype=torch.float32)
    new_labels = torch.tensor(np.array(action), dtype=torch.float32)
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