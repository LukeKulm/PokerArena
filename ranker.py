import numpy as np
import itertools
import time
from parse_hands import Parser
from universal_card_functions import rank_to_prime, primify
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Ranker():
    """
    Class that provides ranking of hands according to the 7,462 equivalence 
    classes that exist in Texas Hold'em Poker
    (Pre-flop hand rankings are indexed to 7,462)
    """

    def __init__(self, parser): # do I still need hand here
        self.parser = parser
        self.parser.parse()
        self.data = self.parser.table()
        self.preflop = self.parser.get_preflop() # preflop parsed data
        # self.hand = hand
        # self.hand_binary = self.encode_hand(hand)
        self.flushes = self.flush_table()
        self.unique = self.unique_ranks()
        self.allelse = self.all_else()

    def rank(self, hand):
        """
        Get the rank of a hand
        Returns int
        """
        hand_binary = self.encode_hand(hand)
        if len(hand) == 2:
            return self.preflop_rank(hand) # NON-ENCODED
        elif len(hand) == 5:
            # TODO: call is_flush
            index = self.bitwise_value(hand_binary[0]) | self.bitwise_value(hand_binary[1]) | self.bitwise_value(hand_binary[2]) | self.bitwise_value(hand_binary[3]) | self.bitwise_value(hand_binary[4])
            if self.is_flush(hand_binary):
                return int(self.flushes[index][0]) # need to do this [0] crap for deprecation warning
            elif self.is_unique(hand_binary):
                return int(self.unique[index][0])
            else:
                res = np.where(self.allelse == self.primefactor(hand_binary))[0]
                return int(res)
        elif len(hand) > 5 and len(hand) <= 7:
            combos = list(itertools.combinations(hand, 5)) #  every 5-card combination of the 6-or-7-card hand
            backer = []
            for combo in combos:
                np_combo = np.array(combo)
                backer.append(self.rank(np_combo))
            return min(backer) # return the highest rank
        elif len(hand) > 7:
            return -1

    def encode_hand(self, hand):
        """
        Encode hand as binary vector
        xxxV VVVV VVVV VVVV SSSS xxPP PPPP 
        Where:
        V = value (rank) of card one-hot
        S = suit of card one-hot
        P = prime number associated with card rank

        HAND COMES IN AS [[value, suit], [value, suit], ...]
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
            p = rank_to_prime(hand[i, 0])
            hand_binary[i] = (v << 12) | (s << 8) | p
        return hand_binary

    def preflop_rank(self, hand):
        """
        Get the rank of a preflop hand
        Precondition: len(self.hand) == 2
        """
        if len(hand) != 2:
            return -1 # maybe find a better error return
        # hand = self.encode_hand(hand)
        c1_val = self.parser.rankmap(hand[0, 0])
        c1_suit = hand[0, 1]
        c2_val = self.parser.rankmap(hand[1, 0])
        c2_suit = hand[1, 1]
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
        """
        Creates lookup table for all flushes (5-card same-suit hands)
        """
        flushes = np.zeros((7937, 1), dtype='int64')
        for row in self.data:
           if row[6] == 0:
               index = row[1] | row[2] | row[3] | row[4] | row[5]
               flushes[index] = row[0]
        return flushes

    def unique_ranks(self):
        """
        Creates lookup table for all straights or high card hands (5-card hands with unique ranks)
        Different from flushes because possibility of conflicting indices
        """
        unique = np.zeros((7937, 1), dtype=int)
        for row in self.data:
            if row[6] == 3 or row[6] == 7: # straight or high card
                index = row[1] | row[2] | row[3] | row[4] | row[5]
                unique[index] = row[0]
        return unique
    
    def all_else(self):
        """
        Creates lookup table for all other hands (full house, four of a kind, three of a kind, pairs)
        Uses the fact that all other hands have prime factorizations that are unique
        I.e. the multiplcation of the prime encodings will create a unique value
        """
        allelse = np.zeros((7462, 1), dtype=int)
        for i in range(len(self.data)):
            if self.data[i][6] != 0 and self.data[i][6] != 3 and self.data[i][6] != 7:
                rank_index = self.data[i][0] # THIS MIGHT BREAK ON THE LAST INDEX -- TEST IT
                t1 = primify(self.data[i][1])
                t2 = primify(self.data[i][2])
                t3 = primify(self.data[i][3])
                t4 = primify(self.data[i][4])
                t5 = primify(self.data[i][5])
                temp = np.array([t1, t2, t3, t4, t5])
                allelse[rank_index] = int(np.prod(temp))
            else:
                allelse[i] = 0
        return allelse

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
    
    def is_unique(self, combo):
        """
        Check if a 5-card combination has unique ranks
        """
        if len(combo) != 5:
            return False
        return self.bitwise_value(combo[0]) != self.bitwise_value(combo[1]) != self.bitwise_value(combo[2]) != self.bitwise_value(combo[3]) != self.bitwise_value(combo[4])

    def bitwise_value(self, card):
        """
        Get the value of a card
        """
        return card >> 12
    
    def bitwise_suit(self, card):
        """
        Get the suit of a card
        """
        return card >> 8 & 0b00000000000000001111
    
    def bitwise_prime(self, card):
        """
        Get the prime number of a card
        """
        return card & 0b0000000000000000000011111111
    
    def primefactor(self, combo):
        """
        Get the prime factor of a 5-card combination
        """
        return self.bitwise_prime(combo[0]) * self.bitwise_prime(combo[1]) * self.bitwise_prime(combo[2]) * self.bitwise_prime(combo[3]) * self.bitwise_prime(combo[4])
    
    def save_data_to_file(self, filename):
        """
        Save self.data to a text file
        """
        np.savetxt(filename, self.data, fmt='%d', delimiter=',')    

if __name__ == "__main__":

    ranker = Ranker(Parser())
    print('\n\nDemoing preflop hand ranking.')
    start_time = time.time()
    hand_1 = np.array([[12, 3], [12, 2]])
    print('Preflop hand is: ' + str(hand_1))
    print('Rank is: ' + str(ranker.preflop_rank(hand_1)) + ' out of 169.')
    print("--- %s seconds to rank a preflop hand ---" % (time.time() - start_time))
    input("Press Enter to continue...\n")

    print('Demoing postflop hand ranking.')
    start_time = time.time()
    hand_2 = np.array([[12, 3], [12, 2], [10, 3], [10, 2], [8, 3]])
    print('Postflop hand is: ' + str(hand_2))
    print('Postflop rank is: ' + str(ranker.rank(hand_2)) + ' out of 7462.')
    print("--- %s seconds to rank a postflop hand ---" % (time.time() - start_time))
    input("Press Enter to continue...\n")

    print('Demoing hand ranking with the turn card.')
    start_time = time.time()
    hand_3 = np.array([[12, 3], [12, 2], [10, 3], [10, 2], [8, 3], [10, 1]])
    print('Hand with turn card is: ' + str(hand_3))
    print('Rank is: ' + str(ranker.rank(hand_3)) + ' out of 7462.')
    print("--- %s seconds to rank a hand that has reached the turn ---" % (time.time() - start_time))
    input("Press Enter to continue...\n")

    print('Demoing hand ranking with the river card.')
    start_time = time.time()
    hand_4 = np.array([[12, 3], [12, 2], [10, 3], [10, 2], [8, 3], [10, 1], [10, 0]])
    print('Hand with turn card is: ' + str(hand_4))
    print('Rank is: ' + str(ranker.rank(hand_4)) + ' out of 7462.')
    print("--- %s seconds to rank a hand that has reached the river ---" % (time.time() - start_time))  