import numpy as np

HAND_DATA_PATH = ".\data\hand_equiv_classes\hands"
PREFLOP_DATA_PATH = ".\data\hand_equiv_classes\preflop"
RANK_MAP = {'A': 0b0001000000000000,
            '14':0b0001000000000000, 
            'K': 0b0000100000000000,
            '13':0b0000100000000000, 
            'Q': 0b0000010000000000,
            '12':0b0000010000000000, 
            'J': 0b0000001000000000,
            '11':0b0000001000000000, 
            'T': 0b0000000100000000,
            '10':0b0000000100000000,
            '9': 0b0000000010000000, 
            '8': 0b0000000001000000, 
            '7': 0b0000000000100000,
            '6': 0b0000000000010000, 
            '5': 0b0000000000001000, 
            '4': 0b0000000000000100, 
            '3': 0b0000000000000010, 
            '2': 0b0000000000000001}

class Parser():
    
    """
    Class to parse Texas Hold'em equivalence classes into data structure
    """

    def __init__(self):
        self.data = np.zeros((7462, 7), dtype=int) # np array for 5+ card equivalence classes
        self.preflop = np.zeros((169, 4), dtype=int) # np array for preflop ranges
        self.tick_data = 0

    def encode_rank_onehot(self, rank):
        """
        Encode rank as one-hot vector
        """
        onehot = np.zeros(16) # for hex purposes possibly
        rank_val = RANK_MAP[rank]
        onehot[16 - rank_val] = 1
        return onehot
    
    def parse(self):
        """
        Parse the hand data
        """
        with open(HAND_DATA_PATH, 'r') as file:
            tick = 0
            for line in file:
                cols = line.split()
                self.data[tick, 0] = (int(cols[0]))
                for i in range(5, 10):
                    self.data[tick, i - 4] = RANK_MAP[cols[i]]
                # self.data[self.tick, 6] = cols[10] # this needs to be encoded bc will always be a string
                tick += 1

    def parse_preflop(self):
      """
      Parse the preflop data
      """
      with open(PREFLOP_DATA_PATH, 'r') as file:
          tick = 0
          for line in file:
              cols = line.split()
              hand = cols[0]
              rank = cols[1]
              if len(hand) == 2: # pair
                  self.preflop[tick, 0] = RANK_MAP[hand[0]]
                  self.preflop[tick, 1] = RANK_MAP[hand[1]]
              else:
                  self.preflop[tick, 0] = RANK_MAP[hand[0]]
                  self.preflop[tick, 1] = RANK_MAP[hand[1]]
                  self.preflop[tick, 2] = 1 if hand[2] == 's' else 0
              self.preflop[tick, 3] = int(rank)
              print(self.preflop[tick])
              tick += 1

    def table(self):
        return self.data
    
    def get_preflop(self):
        self.parse_preflop()
        return self.preflop
    
    def rankmap(self, rank):
        """
        Map rank to one-hot vector
        """
        return RANK_MAP[str(rank)]
    
if __name__ == "__main__":
    parser = Parser()
    parser.parse_preflop()