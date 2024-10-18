# interface for players and some instances including "user entry"

from abc import ABC, abstractmethod

class Player(ABC):
    
    # returns a tuple (move, ammount) where move is 0 for a fold, 1 for a check, 2 for a bet
    @abstractmethod
    def act(self, game):
        pass
    def bet(self, ammount):
        self.balance-=ammount
    def win(self, amm):
        self.balance+=amm

class Human(Player):
    def __init__(self, balance):
        self.balance = balance
    def act(self, game):
        move = "x"
        while move not in "fcr":
            move = input("enter f to fold, c to check, r to raise")
        if move == 'f':
            return (0, 0)
        elif move == 'c':
            return (1, 0)
        else:
            amm = 0
            while amm <=0 or amm > self.balance:
                amm = input("how much do you want to raise by")
            return (2, amm)
    