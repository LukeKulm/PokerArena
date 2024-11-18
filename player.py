# interface for players and some instances including "user entry"

import random
from abc import ABC, abstractmethod
import simulate_games
import math
from bc import NN
import torch

PLAYER_TYPES = ["Human", "DataAggregator", "Random", "MonteCarlo"]


class Player(ABC):
    """
    Abstract class for a player in a poker game
    """
    # returns a tuple (move, ammount) where move is 0 for a fold, 1 for a check, 2 for a bet
    @abstractmethod
    def act(self, game):
        pass

    def bet(self, ammount):
        self.balance -= ammount
        self.in_hand_for += ammount

    def win(self, amm):
        self.balance += amm


def get_suit(num):
    """
    Returns the string representation of a suit given a number

    param num: the number of the suit
    """
    if num == 0:
        return "Hearts"
    elif num == 1:
        return "Diamonds"
    elif num == 2:
        return "Clubs"
    else:
        return "Spades"


def get_rank(num):
    """
    Returns the string representation of a rank given a number

    param num: the number of the rank
    """
    if num < 11:
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
    """
    Returns a string representation of a list of cards

    param lis: the list of cards
    """
    ans = ""
    for i in range(0,  len(lis), 2):
        if lis[i] != 0:
            ans += get_rank(lis[i])
            ans += " "
            ans += get_suit(lis[i+1])
            ans += ", "
    return ans


def decode_stage(num):
    """
    Returns the string representation of a stage in a poker hand given a number

    param num: the number of the stage
    """
    if num == 0:
        return "Preflop"
    elif num == 1:
        return "Flop"
    elif num == 2:
        return "Turn"
    else:
        return "River"


class Human(Player):
    """
    Class for a human player in a poker game, which acts using command line input
    """

    def __init__(self, balance):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False

    def act(self, state):
        """
        Returns the move of the human player

        param state: the state of the game
        """
        n = state[0]
        i = state[1]
        hand = decode_cards(state[2:6])
        idealer = state[6]
        stage = decode_stage(state[7])
        in_hand = state[8]
        pot = state[9]
        board = decode_cards(state[10:20])
        stack = state[20]
        bet = state[21] - state[22]

        print("\nIt is player "+str(i)+"'s turn.")
        print("You have "+str(stack)+" chips in your stack.")
        print("The stage of the hand is "+stage)
        if stage != "Preflop":
            print("The board is "+board[:-2]+".")
        print("Your hand is "+hand[:-2]+".")
        print("There is "+str(pot)+" in the pot.")

        if bet == 0:
            move = "x"
            while move not in "fcre":
                move = input(
                    "enter f to fold, c to check, r to raise, or e to end the game now: ")
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                return (1, 0, 0)
            elif move == 'e':
                return (4, 0, 0)
            else:
                amm = 0
                while int(amm) <= 0 or int(amm) > self.balance:
                    amm = int(input("how much do you want to raise by: "))

                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)

        else:
            move = "x"
            print("The total bets requred this round are " +
                  str(state[21])+" and you're already in for "+str(state[22])+" so it is "+str(bet)+" to call.")
            while move not in "fcre":
                move = input(
                    "enter f to fold, c to call, r to raise, or e to end the game now: ")
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                if bet < self.balance:
                    return (1, bet, 0)
                else:
                    self.allin = True
                    return (1, self.balance, 1)
            elif move == 'e':
                return (4, 0, 0)
            else:
                amm = 0
                while int(amm) <= 0 or int(amm) > self.balance or int(amm) < 2*bet:
                    amm = int(input("how much do you want to raise by: "))
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)


class DataAggregator(Player):
    """
    Class for a human player that remembers actions
    """

    def __init__(self, balance):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False
        self.x = []
        self.y = []

    def act(self, state):
        """
        Returns the move of the human player

        param state: the state of the game
        """
        self.x.append(state)
        move = Human.act(self, state)
        self.y.append([move[0], move[1], move[2]])
        return move


class Random(Player):
    """
    Class for a randomly-acting agent in a poker game
    """

    def __init__(self, balance):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False

    def act(self, state):
        """
        Returns the move of the random player

        param state: the state of the game
        """
        if self.balance == 0:
            return (0, 0, 0)
        bet = state[21]-state[22]
        if bet == 0:
            move = random.choice(['c', 'r'])  # randomly choose check or raise
            if move == 'c':
                return (1, 0, 0)
            else:
                # randomly select amount below balance
                amm = random.randint(0, self.balance)
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)

        else:

            # randomly choose one of the three moves
            move = random.choice(['f', 'c', 'r'])
            if move == 'f':
                return (0, 0,  0)
            elif move == 'c':
                if bet < self.balance:
                    return (1, bet, 0)
                else:
                    self.allin = True
                    return (1, self.balance, 1)
            else:
                # randomly select amount below balance
                amm = random.randint(0, self.balance)
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)
                
def round_prediction(n):
    if n<.5:
        return 0
    elif n <1.5:
        return 1
    else:
        return 2
class BCPlayer(Player):
    def __init__(self, balance, number_of_opps):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False
        model = NN()
        self.model = NN()
        state_dict = torch.load('bc_checkpoint.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    def act(self, state):
        n = state[0]
        i = state[1]
        hand = decode_cards(state[2:6])
        idealer = state[6]
        stage = decode_stage(state[7])
        in_hand = state[8]
        pot = state[9]
        board = decode_cards(state[10:20])
        stack = state[20]
        bet = state[21] - state[22]

        state_tensor = torch.from_numpy(state).float()
        prediction = self.model.forward(state_tensor)
        move = prediction[0]
        ammount = prediction[1]
        jam = prediction[2]
        print(move)
        move = round_prediction(move)
        print(prediction)
        print(move)
        if bet == 0:   
            if move == 0: # model predicts a fold when there is no bet
                return (1, 0,  0)
            elif move == 1: # model predicts a check/call
                return (1, 0, 0)
            elif move  == 2: # model predicts a raise
                if ammount >=2: # raise if it's more than a BB
                    if ammount >= self.balance:
                        self.allin = True
                        return (2, self.balance, 1)
                    else:
                        return (2, ammount, 0)
                else: # otherwise check
                    return (1, 0, 0)
        else:
            if move == 0:
                return (0, 0,  0)
            elif move == 1: # model predicts a call
                if bet < self.balance:
                    return (1, bet, 0)
                else:
                    self.allin = True
                    return (1, self.balance, 1)
            elif move  == 2:
                if ammount < abs(bet-ammount): # fold
                    return (0, 0, 0)
                elif ammount < bet*2 and (abs(bet-ammount) < abs((bet*2)-ammount)): # call
                    return (1, ammount, 0)
                else:
                    ammount = max(ammount, 2*bet)
                    if ammount >= self.balance:
                        self.allin = True
                        return (2, self.balance, 1)
                    else:
                        return (2, ammount, 0) 



    

class AIBasedAgent(Player):
    def __init__(self, balance):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False

    def act(self, state):
        bet = state[21] - state[22]
        agent_hand = state[2:6]
        board = state[10:20]
        pass


class MonteCarloAgent(Player):
    """
    Class for an agent that uses Monte Carlo simulation on a hand to determine its move
    """

    def __init__(self, balance, number_of_opps):
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False

        self.number_of_opps = number_of_opps

    def act(self, state):
        """
        Returns the move of the Monte Carlo agent
        """
        bet = state[21] - state[22]
        win_rate = simulate_games.expected_win_rate(
            state[2:6], state[10:20], self.number_of_opps)
        if bet == 0:
            if win_rate > 0.98:
                self.allin = True
                return (2, self.balance, 1)
            elif win_rate < 0.25:
                return (1, 0, 0)
            else:
                bet_amount = math.floor(win_rate*self.balance/2)
                if bet_amount == 0:
                    return (1, 0, 0)
                return (2, bet_amount, 0)
        else:
            if win_rate > 0.98:
                self.allin = True
                if bet >= self.balance:
                    return (1, self.balance, 1)
                else:
                    return (2, self.balance, 1)
            if win_rate < 0.25:
                return (0, 0, 0)
            else:
                if bet >= self.balance:
                    if win_rate > 0.5:
                        self.allin = True
                        return (1, self.balance, 1)
                    else:
                        return (0, 0, 0)
                else:
                    if win_rate < 0.5:
                        return (1, bet, 0)
                    else:
                        bet_amount = math.floor(
                            bet + (self.balance - bet) * win_rate * 0.5)
                        if bet_amount <= bet:
                            return (1, bet, 0)
                        else:
                            return (2, bet_amount, 0)
        return (0, 0, 0)
