# simulates a game of player objects and makes repeated calls to score_hand

Class Hand:
    def __init__(self):
        self.nums = []
        self.suits = []
    def add_card(self, num, suit):
        self.nums+=num
        self.suits+=suit
    
