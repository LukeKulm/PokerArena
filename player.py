# interface for players and some instances including "user entry"

import random
from abc import ABC, abstractmethod
import simulate_games
import math
from ai_models.q_learning_reinforcement_learning_model import PokerQNetwork, DataBuffer

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
            self.allin = True
            return (1, 0, 1)
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


class QLearningAgent(Player):
    def __init__(self, balance, train=True, epsilon=0.1, save_location="rl_model"):
        self.balance = balance
        self.folded = False
        self.allin = False
        # 13 actions total
        # action 0 is fold
        # action 1 is call
        # action 2 is raise by minimum amount
        # action 3 is raise by minimum amount + 10% of remaining_stack
        # action 4 is raise by minimum amount + 20% of remaining_stack
        # action 5 is raise by minimum amount + 30% of remaining_stack
        # action 6 is raise by minimum amount + 40% of remaining_stack
        # action 7 is raise by minimum amount + 50% of remaining_stack
        # action 8 is raise by minimum amount + 60% of remaining_stack
        # action 9 is raise by minimum amount + 70% of remaining_stack
        # action 10 is raise by minimum amount + 80% of remaining_stack
        # action 11 is raise by minimum amount + 90% of remaining_stack
        # action 12 is raise by minimum amount + 100% of remaining_stack
        self.q_network = PokerQNetwork(
            state_space_size=22, action_state_size=13)
        self.buffer = DataBuffer()
        self.prev_state = None
        self.prev_action = None
        self.prev_balance = None
        # for epsilon greedy
        self.epsilon = epsilon

    def act(self, state):
        if self.prev_state is not None:
            self.buffer.add(
                self.prev_state, self.prev_action, self.balance - self.prev_balance, state, False)

        action = self.q_network.get_action(state, self.epsilon)

        self.prev_state = state
        self.prev_action = action
        self.prev_balance = self.balance

        # TODO: fix this here
        # logic here to validate that the action is valid
        # if not valid update the prev_action
        bet = state[21] - state[22]
        if action == 0:
            # fold
            return (0, 0, 0)
        elif action == 1:
            pass

    def train(self):
        # TODO: implement this to be called from action function every so often
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
