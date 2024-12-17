import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from player_action_game_state import PlayerActionGameState
from game import Hand

# Path to the directory containing hand histories
history_dir = 'data/pluribus_unextracted/30'

files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.phh')]

def extract_action(action, min_bet, curr_stack_size):
    """
    Extracts and returns the action as an integer based on a given action string.

    Returns:
        int: The extracted integer action (0-13).
    """
    # 14 actions total
    # action 0 is fold
    # action 1 is call
    # action 2 is raise by minimum amount
    # action 3 is raise by double minimum amount
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
    
    #make sure to remove parts of the string that are left
    action = action[1:-1]

    # Remove unnecessary parts of the input string 
    action_parts = action.split()
    
    # fold
    if 'f' in action_parts:
        return 0
    
    # call
    if 'cc' in action_parts or 'c' in action_parts:
        return 1
    
    # Check for raise actions
    if 'cbr' in action_parts:
        try:
            raise_amount = int(action_parts[-1])  # last part is the raise
        except ValueError:
            raise ValueError(f"Invalid raise amount in action string: {action}")
        
        #not sure if this calculatio works but seems to
        remaining_stack = curr_stack_size - min_bet
        raise_options = [
            min_bet, #Action 2: min bet
            2 * min_bet,  # Action 3: double minimum raise
            2 * min_bet + 0.05 * remaining_stack,  # Action 4
            2 * min_bet + 0.10 * remaining_stack,  # Action 5
            2 * min_bet + 0.15 * remaining_stack,  # Action 6
            2 * min_bet + 0.20 * remaining_stack,  # Action 7
            2 * min_bet + 0.30 * remaining_stack,  # Action 8
            2 * min_bet + 0.40 * remaining_stack,  # Action 9
            2 * min_bet + 0.50 * remaining_stack,  # Action 10
            2 * min_bet + 0.65 * remaining_stack,  # Action 11
            2 * min_bet + 0.80 * remaining_stack,  # Action 12
            remaining_stack #Action 13
        ]
        
        # Find the closest matching action
        min_collector = 10000
        return_int = 2
        #find which one is closest
        for i, threshold in enumerate(raise_options, start=2):
            if abs(raise_amount - threshold) < min_collector:
                min_collector = abs(raise_amount - threshold)
                return_int = i
        return return_int
    
    #TODO figure out if show muck should return anything but I dont think this hurts
    if 'sm' in action_parts:
        return 1

    raise ValueError(f"Unknown action type in action string: {action}")


# Parsing actions with robust handling
def parse_hole_cards(action):
    parts = [p.strip("'\" ") for p in action.split()]  # Normalize split parts
    cards = parts[-1]
    return cards

def parse_community_cards(action):
    parts = [p.strip("'\" ") for p in action.split()]  # Normalize split parts
    cards = parts[-1]
    return [cards[i:i+2] for i in range(0, len(cards), 2)]

def parse_action(action, state, dealer_position):
    parts = [p.strip("'\" ") for p in action.split()]
    action_type = parts[0]
    player_index = -1
    action_label = None

    if action_type.startswith("p"):  # Player actions
        player_index = int(action_type[1]) - 1
        player_action = parts[1]
        action_label = player_action  # Label this action

        if player_action == "f":  # Fold
            state['folded'][player_index] = True
        elif player_action == "cc":  # Check or Call
            call_amount = max(state['current_bets']) - state['current_bets'][player_index]
            state['current_bets'][player_index] += call_amount
            state['pot'] += call_amount
        elif player_action == "cbr":  # Bet or Raise
            bet_amount = int(parts[2])
            state['current_bets'][player_index] += bet_amount
            state['pot'] += bet_amount
        elif player_action == "sm":  # Show/Muck Cards
            shown_cards = state['hole_cards'][player_index]

    elif action_type == "d":  # Dealer actions
        dealer_action = parts[1]
        if dealer_action == "dh":  # Dealt Hole Cards
            player_index = int(parts[2][1]) - 1
            state['hole_cards'][player_index] = parse_hole_cards(action)
        elif dealer_action == "db":  # Dealt Board Cards
            new_cards = parse_community_cards(action)
            for card in new_cards:
                state['community_cards'].add_card(card[0], card[1])

    return state, (player_index, action_label)

# Determine the stage of the game based on community cards
def determine_stage(community_cards):
    num_cards = len(community_cards.get_cards())  # Use Hand's cards attribute
    if num_cards == 0:
        return 0  # Pre-flop
    elif num_cards == 6:
        return 1  # Flop
    elif num_cards == 8:
        return 2  # Turn
    elif num_cards == 10:
        return 3  # River
    return -1

def determine_dealer_position(actions, num_players):
    # Find the first action in the pre-flop round
    for action in actions:
        parts = [p.strip("'\" ") for p in action.split()]
        if parts[0].startswith('p') and parts[1] in ['f', 'c', 'cb', 'br']:  # Pre-flop actions
            first_actor_index = int(parts[0][1]) - 1
            return (first_actor_index - 1) % num_players  # Dealer is the player before

# Function to encode hand history
def encode_hand_history(antes, blinds_or_straddles, min_bet, actions, players, finishing_stacks, starting_stacks):
    num_players = len(players)
    state = {
        'num_players': num_players,
        'hole_cards': [None] * num_players,
        'community_cards': Hand(),
        'current_bets': [0] * num_players,
        'folded': [False] * num_players,
        'pot': sum(antes) + sum(blinds_or_straddles),
    }
    dealer_position = determine_dealer_position(actions, num_players)
    
    # Initialize the encodings dictionary for each player
    encodings = {player: [] for player in range(num_players)}
    
    # Set initial bets from blinds/straddles
    for i, amount in enumerate(blinds_or_straddles):
        if amount > 0:
            state['current_bets'][i] = amount

    for action in actions:
        # Parse action parts
        parts = [p.strip("'\" ") for p in action.split()]
        action_type = parts[0]
        
        # Only process player actions (skip dealer actions)
        if action_type.startswith("p"):
            player_index = int(action_type[1]) - 1
            player_action = parts[1]
            
            # Create a deep copy of the current state before the action
            pre_action_state = PlayerActionGameState(
                num_players=num_players,
                num_players_folded=sum(state['folded']),
                player_index=player_index,
                player_cards=state['hole_cards'][player_index],
                #TODO This should be the starting stack - amount in for right not finishing stack?
                player_balance=starting_stacks[player_index] - state['current_bets'][player_index],
                dealer_position=dealer_position,
                stage=determine_stage(state['community_cards']),
                pot=state['pot'],
                community_cards=state['community_cards'],
                curr_bet=max(state['current_bets']),
                amount_in_for=state['current_bets'][player_index],
            ).encode()
            
            # Store the state-action pair for this player
            #TODO implement extract action:
            curr_stack_size = starting_stacks[player_index] - state['current_bets'][player_index]
            min_amount_bet = min_bet
            extracted_action = extract_action(action, min_amount_bet, curr_stack_size)
            encodings[player_index].append((pre_action_state, extracted_action))
            
            # Update the state based on the action
            if player_action == "f":  # Fold
                state['folded'][player_index] = True
            elif player_action == "cc":  # Check/Call
                call_amount = max(state['current_bets']) - state['current_bets'][player_index]
                state['current_bets'][player_index] += call_amount
                state['pot'] += call_amount
            elif player_action == "cbr":  # Bet/Raise
                bet_amount = int(parts[2])
                state['current_bets'][player_index] += bet_amount
                state['pot'] += bet_amount
        
        elif action_type == "d":  # Dealer
            dealer_action = parts[1]
            if dealer_action == "dh":  # Dealt Hole Cards
                player_index = int(parts[2][1]) - 1
                state['hole_cards'][player_index] = parse_hole_cards(action)
            elif dealer_action == "db":  # Dealt Board Cards
                new_cards = parse_community_cards(action)
                for card in new_cards:
                    state['community_cards'].add_card(card[0], card[1])

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
        elif key == "starting_stacks":
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
    print(file)
    
    encodings = encode_hand_history(
        hand_data['antes'],
        hand_data['blinds_or_straddles'],
        hand_data['min_bet'],
        hand_data['actions'],
        hand_data['players'],
        hand_data['finishing_stacks'],
        hand_data['starting_stacks']
    )
    
    # Write the encodings to a json file
    output_file = os.path.join("data/pluribus_extracted", f"{os.path.splitext(os.path.basename(file))[0]}_encodings.json")
    
    serialized_encodings = {
        str(player_index): [
            {"state": state.tolist(), "action": action}
            for state, action in action_list
        ]
        for player_index, action_list in encodings.items()
    }

    # Save to the output file
    with open(output_file, "w") as f:
        json.dump(serialized_encodings, f, indent=4)

    print(f"Encodings written to {output_file}")
  

    
    