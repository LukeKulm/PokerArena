import numpy as np
import itertools
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
        self.hand_binary = self.encode_hand(hand)
        # self.hand_binary = np.zeros((len(hand),), dtype=int)
        self.flushes = self.flush_table()

    def rank(self, hand):
        """
        Get the rank of a hand
        Pre: hand is already encoded
        Returns int
        """
        if len(hand) == 2:
            return self.preflop_rank()
        elif len(hand) == 5:
            # TODO: call is_flush
            if self.is_flush(hand):
                index = self.bitwise_value(hand[0]) | self.bitwise_value(hand[1]) | self.bitwise_value(hand[2]) | self.bitwise_value(hand[3]) | self.bitwise_value(hand[4])
                return int(self.flushes[index]) # is this cast legal?
            else:
                return -1
            # TODO: call is_straight
            # TODO: call high_card
            pass
        elif len(hand) > 5 and len(hand) <= 7:
            combos = list(itertools.combinations(hand, 5)) #  every 5-card combination of the 6-or-7-card hand
            backer = []
            for combo in combos:
                backer.append(self.rank(combo))
            print(backer)
            return min(backer) # return the highest rank
        elif len(hand) > 7:
            return -1

    def encode_hand(self, hand):
        """
        Encode hand as binary vector
        xxxV VVVV VVVV VVVV SSSS RRRR xxPP PPPP 
        Where:
        V = value of card one-hot
        S = suit of card one-hot
        R = rank of card (0-12)
        P = prime number associated with card rank
        """
        hand_binary = np.zeros((len(hand),), dtype=int)
        for i in range(len(hand)):
            v = self.parser.rankmap(str(hand[i, 0]))
            if hand[i, 1] == 0:
                s = 0b1000
            elif hand[i, 1] == 1:
                s = 0b0100
            elif hand[i, 1] == 2:
                s = 0b0010
            else:
                s = 0b0001
            r = hand[i, 0] - 2
            p = rank_to_prime(hand[i, 0])
            hand_binary[i] = (v << 16) | (s << 12) | (r << 8) | p
        return hand_binary
            # print("{:032b}".format(self.hand_binary[i]))

    def preflop_rank(self, hand):
        """
        Get the rank of a preflop hand
        Precondition: len(self.hand) == 2
        """
        if len(hand) != 2:
            return -1 # maybe find a better error return
        c1_val = self.parser.rankmap(hand[0, 0])
        c1_suit = self.parser.rankmap(hand[0, 1])
        c2_val = self.parser.rankmap(hand[1, 0])
        c2_suit = self.parser.rankmap(hand[1, 1])
        if c1_val == c2_val and c1_suit == c2_suit:
            return -1

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
        return -1
    
    def flush_table(self):
        flushes = np.zeros((7937, 1), dtype=int)
        for row in self.data:
           if row[6] == 0:
               # print(row)
               index = row[1] | row[2] | row[3] | row[4] | row[5]
               flushes[index] = row[0]
        return flushes
        # print(self.data[1598]) # 7 5 4 3 2 flush
        # print(self.flushes[47]) # hand rank of the 7 5 4 3 2 flush, which is 1599 (smallest flush)

    def is_flush(self, combo):
        """
        Check if a 5-card combination is a flush
        """
        if len(combo) != 5:
            return False
        for card in combo:
            if self.bitwise_suit(card) != self.bitwise_suit(combo[0]):
                return False
        return True

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
    # hand = np.array([[12, 3], [11, 3], [10, 2], [9, 3], [8, 3], [7, 3], [6, 3]]) # for testing
    hand = np.array([[12, 3], [11, 3], [10, 3], [9, 3], [8, 3], [7, 3]])
    ranker = Ranker(Parser(), hand)
    binary = ranker.encode_hand(hand)
    print(ranker.rank(binary))