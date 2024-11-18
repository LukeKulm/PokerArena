# simulates a game of player objects and makes repeated calls to score_hand
import player
from score_hands import best_hand_calc
from universal_card_functions import rank_to_num, suit_to_num, num_to_rank, num_to_suite
import numpy as np
import random


class Game:
    """
    Represents the overall poker game
    """

    def __init__(self, players, start=200):
        self.allin = False
        self.user_ended = False
        self.num_players = len(players)
        self.players = []
        for type in players:
            if type == "Human":
                self.players.append(player.Human(start))
            elif type == "DataAggregator":
                self.players.append(player.DataAggregator(start))
            elif type == "Random":
                self.players.append(player.Random(start))
            elif type == "MonteCarlo":
                self.players.append(
                    player.MonteCarloAgent(start, len(players)-1))
            elif type == "BCPlayer":
                self.players.append(
                    player.BCPlayer(start, len(players)-1))
        self.hands = [Hand() for _ in range(self.num_players)]
        self.dealer_position = 0
        self.order = self.gen_order()
        self.community_cards = Hand()
        self.deck = Deck()
        self.pot = 0
        self.current_bet = 0
        self.stage = 0
        self.sb = 1
        self.bb = 2
        self.stacksize = self.bb * 100
        self.folded = [False]*self.num_players
        self.bets = [0]*self.num_players

    def gen_order(self):
        """
        Generates the order of play based on dealer position
        For example, if dealer is 2 and num_players is 4, order is [3, 0, 1, 2]
        """
        result = []
        start = self.dealer_position + 1
        for i in range(start, start + self.num_players):
            result.append(i % self.num_players)
        return result

    def pg(self):
        """
        Checks if all players have matched a raise or folded, and returns True 
        if the hand may advance
        """
        print("checking if pot is good . . . ")
        for i in range(self.num_players):
            if self.folded[i] or self.bets[i] == self.current_bet or self.players[i].allin:
                pass
            elif self.bets[i] < self.current_bet:
                print("player "+str(i)+" has not matched the current bet of " +
                      str(self.current_bet) + " and has bet "+str(self.bets[i]))
                return False
        return True

    def deal_hole_cards(self):
        """
        Deals two cards to each player
        """
        self.folded = [False]*self.num_players
        self.community_cards.wipe()
        self.deck.wipe()
        for hand in self.hands:
            hand.wipe()
            hand.add_card(*self.deck.deal_card())
            hand.add_card(*self.deck.deal_card())

    def deal_flop(self):
        """
        Deals the first three community cards (the flop)
        """
        for _ in range(3):
            self.community_cards.add_card(*self.deck.deal_card())

    def deal_turn(self):
        """
        Deals the fourth community card (the turn)
        """
        self.community_cards.add_card(*self.deck.deal_card())

    def deal_river(self):
        """
        Deals the fifth community card (the river)
        """
        self.community_cards.add_card(*self.deck.deal_card())

    def get_num_bettors(self):
        ans = 0
        for player in self.players:
            if player.balance > 0 and not player.folded:
                ans += 1
        return ans

    def betting_round(self):
        """
        Simulates a betting round where each player can bet, call/check, or fold
        """
        advance = False
        raiser = None
        self.current_bet = 0
        big_in = True
        small_in = True
        self.bets = [0]*self.num_players
        if self.stage == 0:
            big_in = False
            small_in = False
        self.order = self.gen_order()

        while advance == False and not self.allin:
            for i in self.order:
                player = self.players[i]
                if not small_in:
                    self.pot += 1
                    self.bets[i] += 1
                    player.bet(1)
                    self.current_bet = 1
                    raiser = i
                    print(f"Player {i} pays the Small blind")
                    small_in = True
                    continue
                elif not big_in:
                    self.pot += 2
                    self.bets[i] += 2
                    player.bet(2)
                    self.current_bet = 2
                    raiser = i
                    print(f"Player {i} pays the Big blind")
                    big_in = True
                    continue
                self.win_check()

                if self.folded[i] or player.allin or i == raiser:
                    continue
                state = self.encode(i)

                action, bet_amount,  allin = player.act(state)
                if action == 4:  # end the game
                    print("Game over!")
                    self.user_ended = True
                    return
                if action == 2:  # raise
                    self.pot += bet_amount
                    self.bets[i] += bet_amount
                    player.bet(bet_amount)
                    self.current_bet = self.bets[i]
                    raiser = i
                    print(f"player {i} bets {bet_amount}")
                if action == 0:  # fold
                    self.folded[i] = True
                    print(f"player {i} folds.")
                    if self.win_check():
                        advance = True
                        break
                if action == 1:  # check/call
                    self.pot += bet_amount
                    self.bets[i] += bet_amount
                    player.bet(bet_amount)
                    if bet_amount == 0:
                        print(f"player checks")
                    else:
                        print(f"player calls")
                self.win_check()
            if not advance:
                advance = self.pg()
        self.reset_bets()
        if self.get_num_bettors() < 2:
            self.allin = True

    def reset_bets(self):
        for i in range(self.num_players):
            self.bets[i] = 0

    def step(self):
        """
        Moves the game forward through the stages of a single poker hand
        """
        # Pre-flop: deal hole cards and start betting
        
        for player in self.players:
            player.in_hand_for = 0
            player.allin = False
        self.allin = False
        self.over = False
        self.pot = 0
        print("Dealing hole cards...")
        self.deal_hole_cards()
        self.stage = 0
        self.betting_round()
        if self.over or self.user_ended:
            return
        # Flop: deal first three community cards
        print("Dealing the flop...")
        self.stage = 1
        self.deal_flop()
        print(f"Community cards: {self.community_cards.get_cards()}")
        if not self.allin:
            self.betting_round()
        if self.over or self.user_ended:
            return
        # Turn: deal fourth community card
        print("Dealing the turn...")
        self.stage = 2
        self.deal_turn()
        print(f"Community cards: {self.community_cards.get_cards()}")
        if not self.allin:
            self.betting_round()
        if self.over or self.user_ended:
            return
        # River: deal fifth community card
        print("Dealing the river...")
        self.stage = 3
        self.deal_river()
        print(f"Community cards: {self.community_cards.get_cards()}")
        if not self.allin:
            self.betting_round()

        self.determine_winner(True)
<<<<<<< HEAD
        
=======
        self.dealer_position += 1
>>>>>>> a7cd79e6d3133bb1ef516ce8fd65b716cff67a9d

    def determine_winner(self, showdown):
        """
        Determines the winner based on the best hand

        param showdown: bool, True if showdown logic should be used (i.e., all
        five community cards have been dealt)
        """
        self.over = True
        if not showdown:
            winning_player_idx = None
            for i in self.order:
                player = self.players[i]
                if self.folded[i]:
                    continue
                else:
                    winning_player_idx = i
                    winning_player = player
            print(
                f"player {winning_player_idx} wins the pot of {self.pot} chips!")
            winning_player.win(self.pot)
            self.pot = 0
        elif showdown:

            while self.pot > 0:
                best_hand = None
                winning_player = None
                winning_player_idx = None
                for i in self.order:
                    player = self.players[i]
                    if self.folded[i]:
                        continue
                    full_hand = self.hands[i].get_cards(
                    ) + self.community_cards.get_cards()
                    best_hand_for_player, hand_score = best_hand_calc(
                        full_hand)
                    if player.in_hand_for > 0 and (best_hand is None or hand_score > best_hand):
                        best_hand = hand_score
                        winning_player_idx = i
                        winning_player = player

                if winning_player:
                    if winning_player.balance != 0:  # simpler case, when there are no side pots
                        print(
                            f"player {winning_player_idx} wins a pot of {self.pot} chips!")
                        winning_player.balance += self.pot
                        self.pot = 0
                    else:  # case where there are side pots
                        side = 0
                        max_win = winning_player.in_hand_for
                        for player in self.players:
                            won_from_player = min(max_win, player.in_hand_for)
                            side += won_from_player
                            self.pot -= won_from_player
                            player.in_hand_for -= won_from_player
                        winning_player.balance += side
                        print(
                            f"player {winning_player_idx} wins a pot of {side}!")

    def win_check(self):
        """
        if only one player remains, they win the pot, otherwise continue to showdown
        """
        if self.count_num_folded() == self.num_players - 1:
            self.determine_winner(False)
            return True

    def encode(self, player_ind):
        """
        Encodes the state of the game for a given player in a numpy array

        param player_ind: int, the index of the player in self.players
        """
        state = np.zeros(23, dtype=int)
        state[0] = self.num_players
        state[1] = player_ind
        cards = self.hands[player_ind].get_cards()
        state[2] = rank_to_num(cards[0])  # first card number
        state[3] = suit_to_num(cards[1])  # first card suit
        state[4] = rank_to_num(cards[2])  # second card number
        state[5] = suit_to_num(cards[3])  # second card suit
        state[6] = self.dealer_position
        state[7] = self.stage
        state[8] = self.count_num_folded()
        state[9] = self.pot
        if self.stage == 0:
            for i in range(10, 20):
                state[i] = 0
        elif self.stage == 1:
            for i in range(10, 16, 2):
                print(self.community_cards.get_cards())
                state[i] = rank_to_num(
                    self.community_cards.get_cards()[i-10])
                state[i +
                      1] = suit_to_num(self.community_cards.get_cards()[i-9])

            for i in range(16, 20):
                state[i] = 0
        elif self.stage == 2:
            for i in range(10, 18, 2):
                state[i] = rank_to_num(
                    self.community_cards.get_cards()[i-10])
                state[i +
                      1] = suit_to_num(self.community_cards.get_cards()[i-9])
            for i in range(18, 20):
                state[i] = 0
        else:
            for i in range(10, 20, 2):
                state[i] = rank_to_num(
                    self.community_cards.get_cards()[i-10])
                state[i +
                      1] = suit_to_num(self.community_cards.get_cards()[i-9])
        state[20] = self.players[player_ind].balance
        state[21] = self.current_bet
        state[22] = self.bets[player_ind]
        return state

    def count_num_folded(self):
        count = 0
        for player in self.folded:
            if player:
                count += 1
        return count


class Deck:
    """
    Represents a standard deck of cards for dealing
    """

    def __init__(self):
        self.deck = []
        for suit in "hdcs":
            for num in "23456789TJQKA":
                self.deck.append((num, suit))

    def wipe(self):
        self.deck = []
        for suit in "hdcs":
            for num in "23456789TJQKA":
                self.deck.append((num, suit))

    def deal_card(self):
        i = random.randint(0, len(self.deck)-1)
        # print("there are "+str(len(self.deck))+" cards in the deck and we chose the "+str(i)+" one.")

        old = self.deck
        self.deck = []
        for j in range(len(old)):
            if j != i:
                self.deck.append(old[j])
        return old[i]


class Hand:
    """
    Represents cards that have been dealt, either to the community or to a player
    """

    def __init__(self):
        self.nums = []
        self.suits = []

    def add_card(self, num, suit):
        self.nums.append(num)
        self.suits.append(suit)

    def wipe(self):
        self.nums = []
        self.suits = []

    def get_cards(self):
        cards = []
        for card in list(zip(self.nums, self.suits)):
            cards.append(card[0])
            cards.append(card[1])
        return cards
