# interface for players and some instances including "user entry"

import torch
from bc import NN
import random
from abc import ABC, abstractmethod
import simulate_games
import math
from ai_models.q_learning_reinforcement_learning_model import PokerQNetwork, DataBuffer, train_q_network
import numpy as np
import improve_dataset

PLAYER_TYPES = ["Human", "DataAggregator",
                "Random", "MonteCarlo", "BCPlayer", "QLearningAgent", "SmartBCPlayer"]
PLAYER_TYPES_THAT_REQUIRE_TORCH_MODELS = set(["QLearningAgent"])


class Player(ABC):
    """
    Abstract class for a player in a poker game
    """
    # returns a tuple (move, amount) where move is 0 for a fold, 1 for a check, 2 for a bet
    @abstractmethod
    def act(self, game):
        pass

    def bet(self, amount):
        self.balance -= amount
        self.in_hand_for += amount

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
        self.folds = 0
        self.in_hand_for = 0
        self.balance = balance
        self.folded = False
        self.allin = False

    def act(self, state):
        """
        Returns the move of the random player

        param state: the state of the game
        """
        if self.balance <= 0:
            if state[7] == 0:
                return (0, 0, 0)
            self.allin = True
            return (1, 0, 1)
        bet = state[21]-state[22]
        if bet == 0:
            move = random.choice(['c', 'r'])  # randomly choose check or raise
            if move == 'c':
                return (1, 0, 0)
            else:
                # randomly select amount below balance
                """ amm = random.randint(0, self.balance) """
                amm = (2+np.random.geometric(.5))
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)

        else:

            # randomly choose one of the three moves
            move = random.choice(['f', 'c', 'r'])
            if move == 'f':
                self.folds += 1
                return (0, 0,  0)
            elif move == 'c':
                if bet < self.balance:
                    return (1, bet, 0)
                else:
                    self.allin = True
                    return (1, self.balance, 1)
            else:
                # randomly select amount below balance
                # amm = random.randint(round(bet), self.balance)
                amm = 2+np.random.geometric(.5)
                if amm < bet:
                    amm = bet
                if amm < self.balance:
                    return (2, amm, 0)
                else:
                    self.allin = True
                    return (2, self.balance, 1)


class QLearningAgent(Player):
    def __init__(self, balance, model_path=None, epsilon=0.01, train=False, learn_frequency=1, batch_size=50):
        self.balance = balance
        self.folded = False
        self.allin = False
        # 13 actions total
        # action 0 is fold
        # action 1 is call
        # action 2 is raise by minimum amount
        # action 3 is raise by double minimum amount of remaining_stack
        # action 4 is raise by double minimum amount + 5% of remaining_stack
        # action 5 is raise by double minimum amount + 10% of remaining_stack
        # action 6 is raise by double minimum amount + 15% of remaining_stack
        # action 7 is raise by double minimum amount + 20% of remaining_stack
        # action 8 is raise by double minimum amount + 30% of remaining_stack
        # action 9 is raise by double minimum amount + 40% of remaining_stack
        # action 10 is raise by double minimum amount + 50% of remaining_stack
        # action 11 is raise by double minimum amount + 65% of remaining_stack
        # action 12 is raise by double minimum amount + 80% of remaining_stack
        # action 13 is all in
        self.q_network = PokerQNetwork(
            state_space_size=23, action_space_size=14)
        self.q_network.load_state_dict(torch.load(model_path))
        self.prev_state = None
        self.prev_action = None
        self.prev_balance = None
        # for epsilon greedy
        self.epsilon = epsilon
        self.action_to_additional_percentage = {
            3: 0, 4: 0.05, 5: 0.1, 6: 0.15, 7: 0.2, 8: 0.3, 9: 0.4, 10: 0.5, 11: 0.65, 12: 0.8, 13: 1.0}
        # training_information
        self.buffer = DataBuffer()
        self.train = train
        self.learn_frequency = learn_frequency
        self.batch_size = batch_size
        self.iteration = 0

    def bet_by_double_min_bet_plus_percentage(self, bet, additional_percentage):
        double_min_bet = 4
        if bet != 0:
            double_min_bet = 3 * bet

        if double_min_bet >= self.balance:
            self.allin = True
            return (2, self.balance, 1)

        this_round_bet = double_min_bet + \
            int(np.floor((self.balance - double_min_bet) * additional_percentage))

        if this_round_bet == self.balance:
            self.allin = True
            return (2, this_round_bet, 1)
        else:
            return (2, this_round_bet, 0)

    def get_action_train_and_add_to_buffer(self, state):
        if self.prev_state is not None and self.train:
            self.buffer.add(
                self.prev_state, self.prev_action, self.balance - self.prev_balance, state)
        if self.train:
            if self.iteration % self.learn_frequency == 0:
                self.iteration = 0
                self.train_on_buffer_data()
            else:
                self.iteration += 1

        action = self.q_network.get_action(state, self.epsilon)

        self.prev_state = state
        self.prev_action = action
        self.prev_balance = self.balance

        return action

    def act(self, state):
        action = self.get_action_train_and_add_to_buffer(state)

        bet = state[21] - state[22]
        if action == 0:
            # fold
            return (0, 0, 0)
        if bet >= self.balance:
            # if bet is larger than AI balance, go all in after this point
            self.allin = True
            return (1, self.balance, 1)
        if action == 1:
            return (1, bet, 0)
        elif action == 2:
            min_bet = 2
            if bet != 0:
                min_bet = 2*bet
            if min_bet >= self.balance:
                self.allin = True
                return (2, self.balance, 1)
            else:
                return (2, min_bet, 0)
        else:
            return self.bet_by_double_min_bet_plus_percentage(
                bet, self.action_to_additional_percentage[action])

    def train_on_buffer_data(self):
        # TODO: implement this to be called from action function every so often
        train_q_network(self.q_network, self.buffer,
                        batch_size=self.batch_size)


class MonteCarloQLeaningHybrid:
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


def round_prediction(n):
    if n < .5:
        return 0
    elif n <= 2:
        return 1
    else:
        return 2


class BCPlayer(Player):
    def __init__(self, balance, number_of_opps):
        self.in_hand_for = 0
        self.folds = 0
        self.balance = balance
        self.folded = False
        self.allin = False
        self.model = NN(input_size=23)
        state_dict = torch.load('bc_checkpoint.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.calls = 0
        self.raises = 0

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
        amount = prediction[1].item()
        jam = prediction[2]
        print(move)
        move = round_prediction(move)
        return self.move_from_prediction(bet, move, amount)

        # Use for debugging/explainability:
        # print(prediction)
        # print(amount)
        # print(move)

    def move_from_prediction(self, bet, move, amount):
        if bet == 0:
            if move == 0:  # model predicts a fold when there is no bet
                self.calls += 1
                return (1, 0,  0)
            elif move == 1:  # model predicts a check/call
                self.calls += 1
                return (1, 0, 0)
            elif move == 2:  # model predicts a raise
                if amount >= 2:  # raise if it's more than a BB
                    if amount >= self.balance:
                        self.raises += 1
                        self.allin = True
                        return (2, self.balance, 1)
                    else:
                        self.raises += 1
                        return (2, amount, 0)
                else:  # otherwise check
                    self.calls += 1
                    return (1, 0, 0)
        else:
            if move == 0:
                self.folds += 1
                return (0, 0,  0)
            elif move == 1:  # model predicts a call
                if amount < abs(bet-amount):  # fold
                    self.folds += 1
                    return (0, 0, 0)
                if True:  # call
                    if amount >= self.balance:
                        self.allin = True
                        self.calls += 1
                        return (1, self.balance, 1)
                    else:
                        self.calls += 1
                        return (1, bet, 0)
            elif move == 2:
                if amount < abs(bet-amount):  # fold
                    self.folds += 1
                    return (0, 0, 0)
                # and (abs(bet-amount) < abs((bet*2)-amount)):  # call
                if amount < bet*2:
                    self.calls += 1
                    if bet >= self.balance:
                        self.allin = True
                        return (1, self.balance, 1)
                    else:
                        return (1, bet, 0)

                else:
                    amount = max(amount, 2*bet)
                    if amount >= self.balance:
                        self.allin = True
                        self.raises += 1
                        return (2, self.balance, 1)
                    else:
                        self.raises += 1
                        return (2, amount, 0)


class SmartBCPlayer(BCPlayer):
    def __init__(self, balance, number_of_opps):
        super().__init__(balance, number_of_opps)
        self.model = NN(input_size=11)
        state_dict = torch.load('smart_bc_checkpoint.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, state):
        bet = state[21] - state[22]
        state_tensor = improve_dataset.make_new_state(state)
        # state_tensor = torch.from_numpy(improved_state).float()
        prediction = self.model.forward(state_tensor)
        move = prediction[0]
        amount = prediction[1].item()
        jam = prediction[2]
        print(move)
        if bet == 0:
            if amount >= 2:  # raise if it's more than a BB
                if amount >= self.balance:
                    self.raises += 1
                    self.allin = True
                    return (2, self.balance, 1)
                else:
                    self.raises += 1
                    return (2, amount, 0)
            else:  # otherwise check
                self.calls += 1
                return (1, 0, 0)
        else:
            if amount < abs(bet-amount):  # fold
                self.folds += 1
                return (0, 0, 0)
            if amount < bet*2:  # and (abs(bet-amount) < abs((bet*2)-amount)):  # call
                self.calls += 1
                if bet >= self.balance:
                    self.allin = True
                    return (1, self.balance, 1)
                else:
                    return (1, bet, 0)
            else:
                amount = max(amount, 2*bet)
                if amount >= self.balance:
                    self.allin = True
                    self.raises += 1
                    return (2, self.balance, 1)
                else:
                    self.raises += 1
                    return (2, amount, 0)
                
class PokerTheoryQAgent(QLearningAgent):
    def __init__(self, balance, ranker, model_path = None, epsilon = 0.01, train = False, 
                 learn_frequency = 1, batch_size = 50, eps_decay = 0.999, eps_floor = 0.01): # try different epsilon values
        super().__init__(balance=balance, model_path=model_path, epsilon=epsilon, train=train,
                          learn_frequency=learn_frequency, batch_size=batch_size)
        self.q_network = PokerQNetwork(state_space_size = 25, action_space_size = 14)
        if model_path:
            self.q_network.load_state_dict(torch.load(model_path))

        # loading in ranker for hand evaluation
        self.ranker = ranker

        # trying epsilon-decreasing, assuming convergence on good policy
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_floor = eps_floor

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_floor)

    def rank_state(self, state):
        hole_card_1 = state[2:4]
        hole_card_2 = state[4:6]
        hole_cards = np.array([hole_card_1, hole_card_2])
        stage = state[7]
        rank_preflop = self.ranker.preflop_rank(hole_cards) # preflop rank
        if stage == 0:
            # no post-flop rank
            rank = 0 # should this be 0 or -1????
        if stage == 1:
            # rank 5-card hand
            board = np.append(hole_cards, state[10:16])
            rank = self.ranker.rank(board)
        elif stage == 2:
            # rank 6-card hand
            board = np.append(hole_cards, state[10:18]) # 6-card board
            rank = self.ranker.rank(board)
        else:
            # rank 7-card hand
            board = np.append(hole_cards, state[10:20]) # 7-card board
            rank = self.ranker.rank(board)
        return np.append(state, [rank_preflop, rank])
    
    def get_action_train_and_add_to_buffer(self, state):
            state = self.rank_state(state) # only change from superclass
            if self.prev_state is not None and self.train:
                self.buffer.add(
                    self.prev_state, self.prev_action, self.balance - self.prev_balance, state)
            if self.train:
                if self.iteration % self.learn_frequency == 0:
                    self.iteration = 0
                    self.train_on_buffer_data()
                else:
                    self.iteration += 1

            action = self.q_network.get_action(state, self.epsilon)

            self.prev_state = state
            self.prev_action = action
            self.prev_balance = self.balance

            return action

    def act(self, state):
        state = self.rank_state(state) # only change from superclass
        action = self.get_action_train_and_add_to_buffer(state)

        bet = state[21] - state[22]
        if action == 0:
            # fold
            return (0, 0, 0)
        if bet >= self.balance:
            # if bet is larger than AI balance, go all in after this point
            self.allin = True
            return (1, self.balance, 1)
        if action == 1:
            return (1, bet, 0)
        elif action == 2:
            min_bet = 2
            if bet != 0:
                min_bet = 2*bet
            if min_bet >= self.balance:
                self.allin = True
                return (2, self.balance, 1)
            else:
                return (2, min_bet, 0)
        else:
            return self.bet_by_double_min_bet_plus_percentage(
                bet, self.action_to_additional_percentage[action])
        

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
