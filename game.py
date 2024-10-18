# simulates a game of player objects and makes repeated calls to score_hand
from player import Player
from score_hands import best_hand_calc

class Game:
    """Represents the overall poker game."""
    def __init__(self, num_players):
        self.num_players = num_players
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.dealer_position = 0
        self.community_cards = Hand()
        self.deck = Deck()
        self.pot = 0
        self.current_bet = 0

    def deal_hole_cards(self):
        """Deals two cards to each player."""
        for player in self.players:
            player.hand.add_card(*self.deck.deal_card())
            player.hand.add_card(*self.deck.deal_card())

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
        """Simulates a betting round where each player can bet, call, raise, or fold."""
        for player in self.players:
            if player.is_folded:
                continue
            # For simplicity, each player will bet a fixed amount for now
            try:
                bet_amount = min(player.chips, 50)  # Placeholder for actual betting strategy
                self.pot += player.bet(bet_amount)
                print(f"{player.name} bets {bet_amount}")
            except ValueError as e:
                print(e)
                player.fold()
                print(f"{player.name} folds.")

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
        self.deal_flop()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()
        
        # Turn: deal fourth community card
        print("Dealing the turn...")
        self.deal_turn()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()
        
        # River: deal fifth community card
        print("Dealing the river...")
        self.deal_river()
        print(f"Community cards: {self.community_cards.get_cards()}")
        self.betting_round()

        # At the end, we would call a function to determine the winner based on hand strength
        self.determine_winner()
        
    def determine_winner(self):
        """Determines the winner based on the best hand."""
        best_hand = None
        winning_player = None

        for player in self.players:
            if player.is_folded:
                continue
            full_hand = player.hand.get_cards() + self.community_cards.get_cards()
            best_hand_for_player, hand_score = best_hand_calc(full_hand)
            if best_hand is None or hand_score > best_hand:
                best_hand = hand_score
                winning_player = player
                print(f"{winning_player.name} wins the pot of {self.pot} chips!")
        winning_player.chips += self.pot
        self.pot = 0
    
class Deck:
    def __init__(self):
        self.cards_available = None
        

class Hand:
    def __init__(self):
        self.nums = []
        self.suits = []
    def add_card(self, num, suit):
        self.nums+=num
        self.suits+=suit
    def get_cards(self):
        return list(zip(self.nums, self.suits))
    
