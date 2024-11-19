import numpy as np

HAND_DATA_PATH = ".\data\hand_equiv_classes\hands"
RANK_MAP = {'A': 0b0001000000000000, 
            'K': 0b0000100000000000, 
            'Q': 0b0000010000000000, 
            'J': 0b0000001000000000, 
            'T': 0b0000000100000000, 
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
        self.data = np.zeros((7462, 7), dtype=int)
        self.tick = 0

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
            for line in file:
                # row = []
                cols = line.split()
                self.data[self.tick, 0] = (int(cols[0]))
                for i in range(5, 10):
                    self.data[self.tick, i - 4] = RANK_MAP[cols[i]]
                # self.data[self.tick, 6] = cols[10]
                print(self.data[self.tick])
                self.tick += 1
                return self.data

if __name__ == "__main__":
    parser = Parser()
    parser.parse()