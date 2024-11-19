import numpy as np
from player_action_game_state import PlayerActionGameState
import importlib
import os

# Path to the directory containing hand histories
history_dir = 'hand_histories'

files = [f for f in os.listdir(history_dir) if f.endswith('.phh')]

#parsing actions
def parse_hole_cards(action):
    parts = action.split()
    cards = parts[-1]
    return [cards[:2], cards[2:]]

def parse_action(action, state):
    parts = action.split()
    player_index = int(parts[1][1]) - 1  # pX -> player index
    action_type = parts[2]
    if action_type == "f":  # Fold
        state['folded'][player_index] = True
    elif action_type in ["cb", "br"]:  # Bet or Raise
        bet_amount = int(parts[3])
        state['current_bets'][player_index] += bet_amount
        state['pot'] += bet_amount
    elif action_type.startswith('dh'):  # Dealt Hole cards
        state['hole_cards'][player_index] = parse_hole_cards(action)
        
    #need to get COMMUNITY CARDS

# Function to encode hand history
def encode_hand_history(antes, blinds_or_straddles, min_bet, actions, players, finishing_stacks):
    num_players = len(players)
    #Initialize state
    state = {
        'num_players': num_players,
        'hole_cards': [None] * num_players,
        'community_cards': [],
        'current_bets': [0] * num_players,
        'folded': [False] * num_players,
        'pot': sum(antes) + sum(blinds_or_straddles)
    }

    # Process actions
    for action in actions:
        parse_action(action, state)

    # Encode for each player
    encodings = []
    for i in range(num_players):
        player_state = PlayerActionGameState(
            num_players=num_players,
            num_players_folded=sum(state['folded']),
            player_index=i,
            player_cards=state['hole_cards'][i],
            player_balance=finishing_stacks[i],
            dealer_position=0,  # Assuming dealer is player 0, Need to change
            stage=3,  # Need to Change (update based on parsing logic)
            pot=state['pot'],
            community_cards=state['community_cards'],
            curr_bet=min_bet,
            amount_in_for=state['current_bets'][i]
        )
        encodings.append(player_state.encode())
    return encodings

# Iterate through each file and import its contents
for file in files:
    module_name = f"{history_dir}.{file[:-4]}"  # Remove .phh extension
    module = importlib.import_module(module_name)

    encodings = encode_hand_history(module.antes,
        module.blinds_or_straddles, module.min_bet,
        module.actions, module.players, module.finishing_stacks)
    
    