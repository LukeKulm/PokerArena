import numpy as np
from player import decode_stage
from universal_card_functions import rank_to_num, suit_to_num, num_to_rank, num_to_suite


class PlayerActionGameState:
    """
    Encodes the state of the game for a given player in a numpy array
    """

    def __init__(self, num_players, num_players_folded, player_index,
                 player_cards, player_balance, dealer_position, stage, pot, community_cards,
                 curr_bet, amount_in_for):
        # info about number of players still in the game
        self.num_players = num_players
        self.num_players_folded = num_players_folded

        # info about the player, their cards, and their balance
        self.player_index = player_index
        self.player_cards = player_cards
        self.player_balance = player_balance

        self.dealer_position = dealer_position

        # info about the board, pots, and game stage
        self.stage = stage
        self.pot = pot
        self.community_cards = community_cards

        # info about the bets for this stage
        self.curr_bet = curr_bet
        self.amount_in_for = amount_in_for

    def encode(self):
        """
        Encodes the state of the game for a given player in a numpy array

        param player_ind: int, the index of the player in self.players
        """
        state = np.zeros(23, dtype=int)
        state[0] = self.num_players
        state[1] = self.player_index

        # player cards
        state[2] = rank_to_num(self.player_cards[0])  # first card number
        state[3] = suit_to_num(self.player_cards[1])  # first card suit
        state[4] = rank_to_num(self.player_cards[2])  # second card number
        state[5] = suit_to_num(self.player_cards[3])  # second card suit

        state[6] = self.dealer_position
        state[7] = self.stage
        state[8] = self.num_players_folded
        state[9] = self.pot
        print("Community cards: ", self.community_cards.get_cards())
        print(self.stage)

        if self.stage == 0:
            for i in range(10, 20):
                state[i] = 0
        elif self.stage == 1:
            for i in range(10, 16, 2):
                print(i - 10)
                state[i] = rank_to_num(self.community_cards.get_cards()[i-10])
                state[i+1] = suit_to_num(self.community_cards.get_cards()[i-9])

            for i in range(16, 20):
                state[i] = 0
        elif self.stage == 2:
            for i in range(10, 18, 2):
                state[i] = rank_to_num(
                    self.community_cards.get_cards()[i-10])
                state[i + 1] = suit_to_num(self.community_cards.get_cards()[i-9])
            for i in range(18, 20):
                state[i] = 0
        else:
            for i in range(10, 20, 2):
                state[i] = rank_to_num(
                    self.community_cards.get_cards()[i-10])
                state[i +
                      1] = suit_to_num(self.community_cards.get_cards()[i-9])

        state[20] = self.player_balance
        state[21] = self.curr_bet
        state[22] = self.amount_in_for

        return state

    @property
    def min_bet(self):
        return self.curr_bet - self.amount_in_for
