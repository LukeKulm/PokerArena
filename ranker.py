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
        self.preflop = self.parser.get_preflop() # preflop parsed data
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
        c1_val = self.parser.rankmap(self.hand[0, 0])
        c1_suit = self.parser.rankmap(self.hand[0, 1])
        c2_val = self.parser.rankmap(self.hand[1, 0])
        c2_suit = self.parser.rankmap(self.hand[1, 1])

        if c1_val == c2_val:
            for line in self.preflop:
                if line[0] == c1_val and line[1] == c2_val:
                    return line[3]
        else:
            for line in self.preflop:
                if ((line[0] == c1_val and line[1] == c2_val) or (line[0] == c2_val and line[1] == c1_val)) and c1_suit == c2_suit and line[2] == 1:
                    return line[3]
                elif ((line[0] == c1_val and line[1] == c2_val) or (line[0] == c2_val and line[1] == c1_val)) and c1_suit != c2_suit and line[2] == 0:
                    return line[3]
        return -1 # maybe find a better error return

    def bitwise_value(self, card):
        """
        Get the value of a card
        """
        return card >> 16
    
    def bitwise_suit(self, card):
        """
        Get the suit of a card
        """
        return card >> 12 & 0b00000000000000001111
    
    def bitwise_rank(self, card):
        """
        Get the rank of a card
        """
        return card >> 8 & 0b000000000000000000001111
    
    def bitwise_prime(self, card):
        """
        Get the prime number of a card
        """
        return card & 0b00000000000000000000000011111111

if __name__ == "__main__":
    hand = np.array([[14, 3], [9, 2]])
    ranker = Ranker(Parser(), hand)
    ranker.encode_hand()
    rank = ranker.preflop_rank()
    print(rank)