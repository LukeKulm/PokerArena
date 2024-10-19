# simulates a game of player objects and makes repeated calls to score_hand
import player
from score_hands import best_hand_calc
import numpy as np
import random

class Game:
    """Represents the overall poker game."""
    def __init__(self, players, start=200):
        self.num_players = len(players)
        # self.players = [Player(f"Player {i+1}") for i in range(num_players)] #  wrong
        self.players = []
        for type in players:
            if type == "Human":
                self.players.append(player.Human(start))
            elif type == "Random":
                self.players.append(player.Random(start))
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
        """generates the order of play based on dealer position""" # for example if dealer is 2 and num_players is 4, order is [3, 0, 1, 2]
        result = []
        start = self.dealer_position + 1
        for i in range(start, start + self.num_players):
            result.append(i % self.num_players)
        return result
    
    # def pot_good(self, calls):
    #     for i in range(self.num_players):
    #         if self.folded[i] or self.current_bet == calls[i] or self.players[i].allin:
    #             pass
    #         else:
    #             return False
    #     return True
    
    def pg(self): # missing allin functionality maybe????
        # if all players have either folded or bet the same amount, return True
        print("checking if pot is good . . . ")
        for i in range(self.num_players):
            if self.folded[i] or self.bets[i] == self.current_bet or self.players[i].allin:
                pass
            elif self.bets[i] < self.current_bet:
                print("player "+str(i)+" has not matched the current bet of " +str(self.current_bet)+ " and has bet "+str(self.bets[i]))
                return False
        return True

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
        
    def betting_round(self):
        """Simulates a betting round where each player can bet, call/check, or fold."""

        advance = False
        raiser = None
        self.current_bet = 0
        big_in = True
        small_in = True
        if self.stage == 0:
            big_in = False
            small_in = False
        while advance == False:
           
            for i in self.order:
                player = self.players[i]
                if not small_in:
                    self.pot += 1
                    self.bets[i] = 1
                    player.bet(1)
                    self.current_bet = 1
                    raiser = i
                    print(f"Small blind")
                    small_in = True
                    continue
                elif not big_in:
                    self.pot += 2
                    self.bets[i] = 2
                    player.bet(2)
                    self.current_bet = 2
                    raiser = i
                    print(f"Big blind")
                    big_in = True
                    continue
                self.win_check()
                
                if self.folded[i] or player.allin or i == raiser:
                    continue
                state = self.encode(i)
                
                action, bet_amount,  allin = player.act(state)

                if action == 2: # raise
                    self.pot += bet_amount
                    self.bets[i] = bet_amount
                    player.bet(bet_amount)
                    self.current_bet = bet_amount
                    raiser = i
                    print(f"player bets {bet_amount}")
                if action == 0: # fold
                    self.folded[i] = True
                    print(f"player folds.")
                if action == 1: # check/call
                    self.pot += bet_amount
                    self.bets[i] = bet_amount
                    player.bet(bet_amount)
                    if bet_amount == 0:
                        print(f"player checks")
                    else:
                        print(f"player calls")
            advance = self.pg()
        self.reset_bets()

    def reset_bets(self):
        for i in range(self.num_players):
            self.bets[i] = 0

    def step(self):
        """Moves the game forward one step through the stages."""
        # Pre-flop: deal hole cards and start betting
        print("Dealing hole cards...")
        self.deal_hole_cards()
        self.stage = 0
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

        self.determine_winner(True)
        
    def determine_winner(self, showdown):
        """Determines the winner based on the best hand."""
        if not showdown:
            for i in self.order:
                player = self.players[i]
                if self.folded[i]:
                    continue
                else:
                    winning_player = player
            print(f"player wins the pot of {self.pot} chips!") # change this to print winning player
        
        elif showdown:
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
                    winning_player = player # this assignemnt doens't work
            print(f"player  wins the pot of {self.pot} chips!") # change this to print winning player
        # winning_player.win(self.pot)
        if winning_player:
            winning_player.balance += self.pot # not tested
        self.pot = 0
        self.dealer_position += 1

    def win_check(self):
        # if only one player remains, they win the pot
        # else, continue to showdown
        if self.count_num_folded() == self.num_players - 1:
            self.determine_winner(False)

    def encode(self, player_ind): # player_ind is an index
        state = np.zeros(22, dtype=int)
        state[0] = self.num_players
        state[1] = player_ind
        cards = self.hands[player_ind].get_cards()
        state[2] = self.rank_to_num(cards[0]) # first card number
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
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10])
                state[i+1] = self.suit_to_num(self.community_cards.get_cards()[i-9])
                
            for i in range(16, 20):
                state[i] = 0
        elif self.stage == 2:
            for i in range(10, 18, 2):
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10])
                state[i+1] = self.suit_to_num(self.community_cards.get_cards()[i-9])
            for i in range(18, 20):
                state[i] = 0
        else:
            for i in range(10, 20, 2):
                state[i] = self.rank_to_num(self.community_cards.get_cards()[i-10])
                state[i+1] = self.suit_to_num(self.community_cards.get_cards()[i-9])
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
        # print("there are "+str(len(self.deck))+" cards in the deck and we chose the "+str(i)+" one.")

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
    
