import numpy as np


class PlayerGameState:
    """
    Encodes the state of the game for a given player in a numpy array
    """

    def __init__(self, game, player_ind):
        state = np.zeros(23, dtype=int)
        state[0] = game.num_players
        state[1] = player_ind
        cards = game.hands[player_ind].get_cards()
        state[2] = game.rank_to_num(cards[0])  # first card number
        state[3] = game.suit_to_num(cards[1])  # first card suit
        state[4] = game.rank_to_num(cards[2])  # second card number
        state[5] = game.suit_to_num(cards[3])  # second card suit
        state[6] = game.dealer_position
        state[7] = game.stage
        state[8] = game.count_num_folded()
        state[9] = game.pot
        if game.stage == 0:
            for i in range(10, 20):
                state[i] = 0
        elif game.stage == 1:
            for i in range(10, 16, 2):
                print(game.community_cards.get_cards())
                state[i] = game.rank_to_num(
                    game.community_cards.get_cards()[i-10])
                state[i +
                      1] = game.suit_to_num(game.community_cards.get_cards()[i-9])

            for i in range(16, 20):
                state[i] = 0
        elif game.stage == 2:
            for i in range(10, 18, 2):
                state[i] = game.rank_to_num(
                    game.community_cards.get_cards()[i-10])
                state[i +
                      1] = game.suit_to_num(game.community_cards.get_cards()[i-9])
            for i in range(18, 20):
                state[i] = 0
        else:
            for i in range(10, 20, 2):
                state[i] = game.rank_to_num(
                    game.community_cards.get_cards()[i-10])
                state[i +
                      1] = game.suit_to_num(game.community_cards.get_cards()[i-9])
        state[20] = game.players[player_ind].balance
        state[21] = game.current_bet
        state[22] = game.bets[player_ind]

        self.state = state

    def get_player_hand_cards(self):
        return self.state[2:6]

    def get_board_cards(self):
        return self.state[10:20]

    def min_to_stay_in(self):
        return self.state[21] - self.state[22]
