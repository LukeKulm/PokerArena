# interface for players and some instances including "user entry"

import random
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
def get_suit(num):
    if num == 0:
        return "Hearts"
    elif num == 1:
        return "Diamonds"
    elif num == 2:
        return "Clubs"
    else:
        return "Spades"
def get_rank(num):
    print(num)
    if num <11:
        return str(num)
    else:
        if num == 11:
            return "Jack"
        elif num == 12:
            return "Queen"
        elif num == 13:
            return "King"
        else:
            return "Ace"
def decode_cards(lis):
    ans = ""
    for i in range(0,  len(lis), 2):
        ans+= get_rank(lis[i])
        ans+=" "
        ans+= get_suit(lis[i+1])
        ans+=", "
    return ans
def decode_stage(num):
    if num == 0:
        return "Preflop"
    elif num == 1:
        return "Flop"
    elif num == 2:
        return "Turn"
    else:
        return "River"

class Human(Player):
    def __init__(self, balance):
        self.balance = balance
        self.folded = False
    def act(self, state):
        n = state[0]
        i  = state[1]
        hand =  decode_cards(state[2:6])
        idealer =  state[6]
        stage =   decode_stage(state[7])
        in_hand  = state[8]
        pot = state[9]
        board = decode_cards(state[10:20])
        stack = state[20]
        bet = state[21]
        # if  i  ==  idealer:
        #     print("You are the dealer.")
        # elif i>idealer:
        #     print("You are in the "+i-idealer+" position.")
        # else:
        #     print("You are in the "+n-(idealer-i)+" position.")

        message = "You have "+str(stack)+" chips in your stack. The stage of the hand is "+stage+", the board is "+board+"and your hand is "+hand+" and there is "+str(pot)+" in the pot."
        print(message)
        if bet == 0:
            move = "x"
            while move not in "fcr":
                move = input("enter f to fold, c to check, r to raise")
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                return (1, 0, 0)
            else:
                amm = 0
                while amm <=0 or amm > self.balance:
                    amm = input("how much do you want to raise by")
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    return  (2, self.balance, 1)
            
        else:
            move = "x"
            while move not in "fcr":
                move = input("enter f to fold, c to call, r to raise")
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                if bet<self.balance:
                    return (1, bet, 0)
                else:
                    return (1, self.balance, 1)
            else:
                amm = 0
                while amm <=0 or amm > self.balance or amm<2*bet:
                    amm = input("how much do you want to raise by")
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    return  (2, self.balance, 1)
                
class Random(Player):
    def __init__(self, balance):
        self.balance = balance
        self.folded = False
    def act(self, state):
        bet = state[21]
        if bet == 0:
            move = random.choice(['c', 'r']) #randomly choose check or raise
            if move == 'c':
                return (1, 0, 0)
            else:
                amm = random.randint(0, self.balance) #randomly select amount below balance
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    return  (2, self.balance, 1)
            
        else:
            move = random.choice(['f', 'c', 'r']) #randomly choose one of the three moves
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                if bet<self.balance:
                    return (1, bet, 0)
                else:
                    return (1, self.balance, 1)
            else:
                amm = random.randint(0, self.balance) #randomly select amount below balance
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    return  (2, self.balance, 1)
