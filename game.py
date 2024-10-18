# simulates a game of player objects and makes repeated calls to score_hand
import player
from score_hands import best_hand_calc
import numpy as np
import random

class Game:
    """Represents the overall poker game."""
    def __init__(self, num_players):
        self.num_players = num_players
        # self.players = [Player(f"Player {i+1}") for i in range(num_players)] #  wrong
        self.players = []
        for i in range(num_players):
            self.players.append(player.Human(200))
        self.hands = [Hand() for _ in range(num_players)]
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
        self.folded = [False]*num_players

    def gen_order(self):
        """generates the order of play based on dealer position""" # for example if dealer is 2 and num_players is 4, order is [3, 0, 1, 2]
        result = []
        start = self.dealer_position + 1
        for i in range(start, start + self.num_players):
            result.append(i % self.num_players)
        return result

    def deal_hole_cards(self):
        """Deals two cards to each player."""
        self.folded = [False]*self.num_players
        for hand in self.hands:
            hand.add_card(*self.deck.deal_card())
            hand.add_card(*self.deck.deal_card())

    def deal_flop(self):
        """Deals the first three community cards (the flop)."""
        for _ in range(3):
            self.community_cards.add_card(*self.deck.deal_card())

    def deal_turn(self):
        """Deals the fourth community card (the turn)."""
        self.community_cards.add_card(*self.deck.deal_card())

    def deal_river(self):
        """Deals the fifth community card (the river)."""
        self.community_cards.add_card(*self.deck.deal_card())

    def betting_round(self): # TODO: change acting order based on self.order
        """Simulates a betting round where each player can bet, call/check, or fold."""
        # start_position = (self.dealer_position + 1) % self.num_players
        # for i in range(0, self.num_players):
        for i in self.order:
            # current_player = (i + start_position) % self.num_players
            # player = self.players[current_player]
            player = self.players[i]
            if self.folded[i]:
                continue
            state = self.encode(i)
            bet_amount, action, allin = player.act(state)
            if action == "b":
                self.pot += bet_amount
                player.bet(bet_amount)
                print(f"{player.name} bets {bet_amount}")
            if action == "f":
                self.folded[i] = True
                print(f"{player.name} folds.")
            if action == "c":
                self.pot += bet_amount
                player.bet(bet_amount)
                if bet_amount == 0:
                    print(f"{player.name} checks")
                else:
                    print(f"{player.name} calls")

    def reset_bets(self):
        """Resets the bets for each player at the end of the betting round."""
        for player in self.players:
            player.reset_bet()

    def step(self):
        """Moves the game forward one step through the stages."""
        # Pre-flop: deal hole cards and start betting
        print("Dealing hole cards...")
        self.deal_hole_cards()
        self.betting_round()
        
        # Flop: deal first three community cards
        print("Dealing the flop...")
        self.stage = 1
        self.deal_flop()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()
        
        # Turn: deal fourth community card
        print("Dealing the turn...")
        self.stage = 2
        self.deal_turn()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()
        
        # River: deal fifth community card
        print("Dealing the river...")
        self.stage = 3
        self.deal_river()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()

        # At the end, we would call a function to determine the winner based on hand strength
        self.determine_winner()
        self.dealer_position += 1
        
    def determine_winner(self):
        """Determines the winner based on the best hand."""
        best_hand = None
        winning_player = None

        for i in self.order:
            player = self.players[i]
            if self.folded[i]:
                continue
            full_hand = self.hands[i].get_cards() + self.community_cards.get_cards()
            best_hand_for_player, hand_score = best_hand_calc(full_hand)
            if best_hand is None or hand_score > best_hand:
                best_hand = hand_score
                winning_player = player
                print(f"{winning_player.name} wins the pot of {self.pot} chips!")
        winning_player.chips += self.pot
        self.pot = 0

    def encode(self, player_ind): # player_ind is an index
        state = np.zeros(22, dtype=int)
        state[0] = self.num_players
        state[1] = player_ind
        cards = self.hands[player_ind].get_cards()
        state[2] = self.rank_to_num(cards[0]) # first card number
        print(self.hands[player_ind].get_cards()[0][0])
        state[3] = self.suit_to_num(cards[1]) # first card suit
        state[4] = self.rank_to_num(cards[2]) # second card number
        state[5] = self.suit_to_num(cards[3]) # second card suit
        state[6] = self.dealer_position
        state[7] = self.stage
        state[8] = self.count_num_folded()
        state[9] = self.pot
        if self.stage == 0:
            for i in range(10, 20):
                state[i] = 0
        elif self.stage == 1:
            for i in range(10, 16, 2):
                print(self.community_cards)
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10][0])
                
            for i in range(16, 20):
                state[i] = 0
        elif self.stage == 2:
            for i in range(10, 18, 2):
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10][0])
            for i in range(18, 20):
                state[i] = 0
        else:
            for i in range(10, 20, 2):
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10][0])
        state[20] = self.players[player_ind].balance
        state[21] = self.current_bet
        return state

    def count_num_folded(self):
        count = 0
        for player in self.folded:
            if player:
                count += 1
        return count
    
    def rank_to_num(self, rank):
        if rank == "T":
            return 10
        if rank == "J":
            return 11
        if rank == "Q":
            return 12
        if rank == "K":
            return 13
        if rank == "A":
            return 14
        else:
            return int(rank)
    
    def suit_to_num(self, suit):
        if suit == "c":
            return 0
        if suit == "d":
            return 1
        if suit == "h":
            return 2
        else: # spade
            return 3

class Deck:
    """Represents a deck of cards for dealing."""
    def __init__(self):
        self.deck = []
        for suit in "hdcs":
            for num in "23456789TJQKA":
                self.deck.append((num, suit))
    
    def deal_card(self):
        i = random.randint(0, len(self.deck)-1)
        print("there are "+str(len(self.deck))+" cards in the deck and we chose the "+str(i)+" one.")

        old = self.deck
        self.deck = []
        for j in range(len(old)):
            if j!=i:
                self.deck.append(old[j])
        return old[i]
        

class Hand:
    def __init__(self):
        self.nums = []
        self.suits = []
    def add_card(self, num, suit):
        self.nums.append(num)
        self.suits.append(suit)
    def get_cards(self):
        cards = []
        for card in list(zip(self.nums, self.suits)):
            cards.append(card[0])
            cards.append(card[1])
        return cards
    
