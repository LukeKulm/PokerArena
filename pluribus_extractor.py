import numpy as np
from player_action_game_state import PlayerActionGameState
import importlib
import os

# Path to the directory containing hand histories
history_dir = 'data/pluribus_unextracted/30'

files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.phh')]
print(files)

# Parsing actions
def parse_hole_cards(action):
    parts = action.split()
    cards = parts[-1]
    print(cards)
    return [cards[:2], cards[2:]]

def parse_community_cards(action):
    parts = action.split()
    cards = parts[-1]
    return [cards[i:i+2] for i in range(0, len(cards), 2)]

def parse_action(action, state, dealer_position):
    print(action)
    parts = action.split()
    action_type = parts[0]  # First part indicates the action type

    if action_type == "p":  # Player actions
        player_index = int(parts[1][1]) - 1
        player_action = parts[2]

        if player_action == "f":  # Fold
            state['folded'][player_index] = True

        elif player_action in ["cb", "br"]:  # Bet or Raise
            bet_amount = int(parts[3])
            state['current_bets'][player_index] += bet_amount
            state['pot'] += bet_amount

        elif player_action == "c":  # Call
            call_amount = max(state['current_bets']) - state['current_bets'][player_index]
            state['current_bets'][player_index] += call_amount
            state['pot'] += call_amount

    elif action_type == "d":  # Dealer actions
        dealer_action = parts[1]
        print(dealer_action)

        if dealer_action == "dh":  # Dealt Hole Cards
            player_index = int(parts[2][1]) - 1
            state['hole_cards'][player_index] = parse_hole_cards(action)

        elif dealer_action == "db":  # Dealt Board Cards
            new_cards = parse_community_cards(action)
            state['community_cards'].extend(new_cards)

    return state


# Determine the stage of the game based on community cards
def determine_stage(community_cards):
    if len(community_cards) == 0:
        return 0  # Pre-flop
    elif len(community_cards) == 3:
        return 1  # Flop
    elif len(community_cards) == 4:
        return 2  # Turn
    elif len(community_cards) == 5:
        return 3  # River
    return -1
  
def determine_dealer_position(action, num_players):
    # Find the first action in the pre-flop round
    if action[0].startswith('p') and action[2] in ['f', 'c', 'cb', 'br']:  # Pre-flop actions
        first_actor_index = int(action[0][1]) - 1
        return (first_actor_index - 1) % num_players  # Dealer is the player before

# Function to encode hand history
def encode_hand_history(antes, blinds_or_straddles, min_bet, actions, players, finishing_stacks):
  num_players = len(players)
  # Initialize state
  state = {
    'num_players': num_players,
    'hole_cards': [None] * num_players,
    'community_cards': [],
    'current_bets': [0] * num_players,
    'folded': [False] * num_players,
    'pot': sum(antes) + sum(blinds_or_straddles)
  }

  for action in actions:
    dealer_position = determine_dealer_position(action, num_players)
    state = parse_action(action, state, dealer_position)

  stage = determine_stage(state['community_cards'])

  # Encode for each player
  encodings = []
  for i in range(num_players):
    print(state['hole_cards'])
    player_state = PlayerActionGameState(
      num_players=num_players,
      num_players_folded=sum(state['folded']),
      player_index=i,
      player_cards=state['hole_cards'][i],
      player_balance=finishing_stacks[i],
      dealer_position=dealer_position,
      stage=stage,
      pot=state['pot'],
      community_cards=state['community_cards'],
      curr_bet=max(state['current_bets']),
      amount_in_for=state['current_bets'][i]
    )
    encodings.append(player_state.encode())
  return encodings


# Function to read a single `.phh` file
def read_hand_history(file_path):
    with open(file_path, 'r') as f:
        # Assuming the file contains the hand history in a structured format
        lines = f.readlines()
    return parse_hand_history(lines)

# Function to parse hand history from lines
def parse_hand_history(lines):
    # Parse the game details from the text
    parsed_data = {}
    for line in lines:
        key, value = line.strip().split(' = ')
        if key == "actions":
            parsed_data[key] = value.strip('[]').split(', ')
        elif key in ["antes", "blinds_or_straddles", "starting_stacks", "finishing_stacks"]:
            parsed_data[key] = list(map(int, value.strip('[]').split(', ')))
        elif key == "players":
            parsed_data[key] = value.strip('[]').split(', ')
        elif key == "variant":
            parsed_data[key] = value.strip("'")
        elif key == "ante_trimming_status":
            parsed_data[key] = value.lower() == "true"
        else:
            parsed_data[key] = int(value)  # Assume integers for other fields
    return parsed_data

for file in files:
    hand_data = read_hand_history(file)
    
    encodings = encode_hand_history(
        hand_data['antes'],
        hand_data['blinds_or_straddles'],
        hand_data['min_bet'],
        hand_data['actions'],
        hand_data['players'],
        hand_data['finishing_stacks']
    )
    
    # Write the encodings to a `.txt` file
    output_file = f"{os.path.splitext(os.path.basename(file))[0]}_encodings.txt"
    with open(output_file, "w") as f:
        for encoding in encodings:
            f.write(str(encoding) + "\n")
    print(f"Encodings written to {output_file}")
  

    
    