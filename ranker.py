import numpy as np
from parse_hands import Parser
from universal_card_functions import rank_to_prime

class Ranker():
    """
    Class that provides ranking of hands according to equivalence classes
    """

    def __init__(self, parser, hand):
        self.parser = parser
        self.parser.parse()
        self.data = self.parser.table()
        self.preflop = self.parser.get_preflop()
        self.hand = hand
        self.hand_binary = np.zeros((len(hand),), dtype=int)

    def encode_hand(self):
        """
        Encode hand as binary vector
        xxxV VVVV VVVV VVVV SSSS RRRR xxPP PPPP 
        Where:
        V = value of card one-hot
        S = suit of card one-hot
        R = rank of card (0-12)
        P = prime number associated with card rank
        """
        for i in range(len(self.hand)):
            v = self.parser.rankmap(str(self.hand[i, 0]))
            if self.hand[i, 1] == 0:
                s = 0b1000
            elif self.hand[i, 1] == 1:
                s = 0b0100
            elif self.hand[i, 1] == 2:
                s = 0b0010
            else:
                s = 0b0001
            r = self.hand[i, 0] - 2
            p = rank_to_prime(self.hand[i, 0])
            self.hand_binary[i] = (v << 16) | (s << 12) | (r << 8) | p
            print("{:032b}".format(self.hand_binary[i]))

    def preflop_rank(self):
        """
        Get the rank of a preflop hand
        Precondition: len(self.hand) == 2
        """
        c1 = self.hand[0]
        c2 = self.hand[1]
        if c1 == c2:
            rank = self.preflop[self.parser.encode_rank_onehot(c1), 3]
        else:
            rank = self.preflop[self.parser.encode_rank_onehot(c1), self.parser.encode_rank_onehot(c2), 2]

if __name__ == "__main__":
    hand = np.array([[9, 3], [10, 3]])
    ranker = Ranker(Parser(), hand)
    ranker.encode_hand()