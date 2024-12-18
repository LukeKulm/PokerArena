import streamlit as st
import numpy as np
import sys
import os
import player
from ranker import Ranker
from parse_hands import Parser

def get_rank(selected):
    d = {"A": 14, "K": 13, "Q": 12, "J": 11}
    if selected in ["A", "K", "Q", "J"]:
        return d[selected]
    else:
        return int(selected)


def main():
    st.title("Poker Odds Calculator")

    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    suit_3 = "Hearts"
    suit_4 = "Hearts"
    suit_5 = "Hearts"
    suit_6 = "Hearts"
    suit_7 = "Hearts"

    # Number of players
    num_players = st.number_input(
        "Number of Players in the Game:", min_value=2, value=3, step=1
    )

    # Number of players still in the hand
    num_players_in_hand = st.number_input(
        "Number of Players Still in the Hand:", min_value=2, max_value=num_players, value=num_players, step=1
    )
    if num_players == 2:
        positions = ["Dealer/Big Blind", "Small Blind"]
    else:
        positions = ["Dealer", "Small Blind", "Big Blind"]
        for i in range(2, num_players):
            positions.append(str(i))
    position = st.selectbox("Your position:", positions)

    # Stage of the game
    stage = st.selectbox(
        "Stage of the Game:", ["Preflop", "Flop", "Turn", "River"]
    )

    # Pot size
    pot = st.number_input("Pot Size:", min_value=3, value=3, step=1)

    # Stack size
    stack = st.number_input("Stack Size:", min_value=0, value=200, step=1)

    # Amount to call
    to_call = st.number_input("To Call:", min_value=0, value=0, step=1)

    st.divider()  # Add a horizontal divider for better UI


    st.subheader("Select Your Hand! Note: no duplicate cards allowed.")
    col1, col2 = st.columns(2)

    # Place widgets in col1
    with col1:
        num_1 = st.selectbox(
            "Card 1 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key="num_1"
        )
        num_2 = st.selectbox(
            "Card 2 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key="num_2"
        )

# Place widgets in col2
    with col2:
        suit_1 = st.selectbox(
            "Card 1 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key="suit_1"
        )
        
        suit_2 = st.selectbox(
            "Card 2 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key="suit_2"
        )
    if stage!="Preflop":
        st.subheader("Select the Board! Note: no duplicate cards allowed.")
        col1, col2 = st.columns(2)
        with col1:
            num_3 = st.selectbox(
                "Board Card 1 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key=4
            )
            num_4 = st.selectbox(
                "Board Card 2 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key=6
            )
            num_5 = st.selectbox(
                "Board Card 3 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key=8
            )
            if stage in ["Turn", "River"]:
                num_6 = st.selectbox(
                "Board Card 4 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key=9
            )
            if stage == "River":
                num_7 = st.selectbox(
                "Board Card 5 - Number:", ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"], key=10
            )
        with col2:
            suit_3 = st.selectbox(
                "Board Card 1 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key=5
            )
            suit_4 = st.selectbox(
                "Board Card 2 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key=7
            )
            suit_5 = st.selectbox(
                "Board Card 3 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key=14
            )
            if stage in ["Turn", "River"]:
                num_6 = st.selectbox(
                "Board Card 4 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key=11)
            if stage == "River":
                num_7 = st.selectbox(
                "Board Card 5 - Suit:", ["Hearts", "Diamonds", "Clubs", "Spades"], key=12)
    if stage in ["Turn", "River"]:
        pass
    if st.button("Calculate"):
        calculate_results(
                num_players, num_players_in_hand, position , 
           num_1, suit_1, num_2, suit_2, num_3, suit_3, num_4, suit_4, num_5, suit_5, num_6, suit_6, num_7, suit_7, 
           stage, pot, stack, to_call
)

def calculate_results(
    num_players, num_players_in_hand, position , 
           num_1, suit_1, num_2, suit_2, num_3, suit_3, num_4, suit_4, num_5, suit_5, num_6, suit_6, num_7, suit_7, 
           stage, pot, balance, to_call
):

    state = np.zeros(23, dtype=int)
    state[0] = num_players
    if position== "Dealer" or position == "Dealer/Big Blind":
        state[1] = 0
    elif position == "Small Blind":
        state[1] = 1
    elif position == "Big Blind":
        state[1] = 2
    else:
        state[1] = int(position)
    suits = {"Clubs":0, "Diamonds":1, "Hearts":2, "Spades":3}
    stages = {"Preflop": 0, "Flop": 1, "Turn": 2, "River": 3}
    state[2] = get_rank(num_1)
    state[3] = suits[suit_1]
    state[4] = get_rank(num_2)
    state[5] = suits[suit_2]
    state[6] = 0
    state[7] = stages[stage]
    state[8] = num_players-num_players_in_hand
    state[9] = pot
    
    state[10] = get_rank(num_3)
    state[11] = suits[suit_3]
    state[12] = get_rank(num_4)
    state[13] = suits[suit_4]
    state[14] = get_rank(num_5)
    state[15] = suits[suit_5]
    state[16] = get_rank(num_6)
    state[17] = suits[suit_6]
    state[18] = get_rank(num_7)
    state[19] = suits[suit_7]
    state[20] = balance
    state[21] = to_call
    
    print(state)

    smart_bc = player.SmartBCPlayer(balance, num_players-1)
    smart_bc_move = smart_bc.act(state)

    bc = player.BCPlayer(balance, num_players-1)
    bc_move = bc.act(state)

    ql = player.QLearningAgent(balance)
    ql_move = ql.act(state)

    mc = player.MonteCarloAgent(balance, num_players-1)
    mc_move = mc.act(state)

    hybrid = player.MonteCarloQLearningHybrid(balance)
    hybrid_move = hybrid.act(state)

    # pt = player.PokerTheoryQAgent(balance, Ranker(Parser(), ), "ai_models\q_learning_reinforcement_learning_model.py")
    # pt_move = pt.act(state)

    st.write("Smart BC Player predicts a ", prediction_to_string(smart_bc_move))
    st.write("BC Player predicts a ", prediction_to_string(bc_move))
    st.write("Q Learning Agent predicts a ", prediction_to_string(ql_move))
    st.write("Monte Carlo Agent predicts a ", prediction_to_string(mc_move))
    st.write("MonteCarlo/Q Learning hybrid predicts a ", prediction_to_string(hybrid_move))
    # st.write("Poker Theory Agent predicts a ", prediction_to_string(pt_move))

def prediction_to_string(move):
    action = move[0]
    amount = move[1]
    allin = move[2]
    if action == 2:
        return f"raise of {amount}!"
    if action == 1 and amount ==0:
        return "check!"
    if action == 1:
        return "call!"
    else:
        return "fold!"

if __name__ == "__main__":
    main()
