# simulates a game of player objects and makes repeated calls to score_hand

class Hand:
    def __init__(self):
        self.nums = []
        self.suits = []
    def add_card(self, num, suit):
        self.nums+=num
        self.suits+=suit
    def get_cards(self):
        return list(zip(self.nums, self.suits))
    
