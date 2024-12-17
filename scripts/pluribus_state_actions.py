import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import json



def extract_moves(state_and_action):
  """gets the moves the player made in the game"""
  states = []
  actions = []
  for entry in state_action_pairs:
    state = entry.get("state")
    action = entry.get("action")
    if state is not None and action is not None:
      states.append(np.array(state, dtype=np.float32))
      actions.append(action)
  return states, actions
  

def aggregate(states, actions):
    """
    Creates .pt files of information from extracted encodings of pluribus games
    """
    # convert to tensors
    new_data = torch.tensor(np.stack(states), dtype=torch.float32)
    new_labels = torch.tensor(np.array(actions), dtype=torch.int64)

    # Check if file already exists
    save_path = 'data/pluribus_policy.pt'
    if not os.path.exists(save_path):
        torch.save((new_data, new_labels), save_path)
        print(f"Saved new data to {save_path}")
    else:
        existing_data, existing_labels = torch.load(save_path)
        # add old to new
        updated_data = torch.cat((existing_data, new_data), dim=0)
        updated_labels = torch.cat((existing_labels, new_labels), dim=0)
        torch.save((updated_data, updated_labels), save_path)
        print(f"Updated data saved to {save_path}")


if __name__ == "__main__":
    hist_dir = "data/pluribus_extracted"
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    
    files = [os.path.join(hist_dir, f) for f in os.listdir(hist_dir) if f.endswith('.json')]
    all_states = []
    all_actions = []

    for file_path in files:
        with open(file_path, "r") as f:
          data = json.load(f)

        # iterate over players
        #TODO for now this is fine but eventually will want just one player
        for player_id, state_action_pairs in data.items():
            states, actions = extract_moves(state_action_pairs)
            all_states.extend(states)
            all_actions.extend(actions)

    # aggregate and save to .pt file
    if all_states and all_actions:
        aggregate(all_states, all_actions)
    else:
        print("No valid state-action pairs found.")