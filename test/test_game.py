import pytest
import game
import player


def test_game_step_random_players():
  """Test that game can run many times with random players"""
  for i in range(1000):
    g = game.Game(["Random", "Random", "Random"], 200)
    g.step()
  
def test_random_vs_montecarlo_players():
  """Test that game can run many times with random vs montecarlo players"""
  montecarlo_wins = 0
  random_wins = 0
  for i in range(100):
    g = game.Game(["Random", "MonteCarlo"], 200)
    g.step()
    if g.players[0].balance > g.players[1].balance:
        montecarlo_wins += 1
    elif g.players[1].balance > g.players[0].balance:
        random_wins += 1
  assert montecarlo_wins > random_wins


@pytest.fixture
def create_game():
  """Fixture to create a game with 3 players."""
  players = ["Human", "Random", "Random"]
  return game.Game(players, start=200)

@pytest.fixture
def create_random_game():
  """Fixture to create a game with 3 random players."""
  players = ["Random", "Random", "Random"]
  return game.Game(players, start=200)

def test_game_initialization(create_game):
    """Test that the game initializes with the correct number of players. Fairly Trivial"""
    game = create_game
    assert game.num_players == 3
    assert len(game.players) == 3
    assert isinstance(game.players[0], player.Human)
    assert isinstance(game.players[1], player.Random)
    assert isinstance(game.players[2], player.Random)
    
def test_deck_dealing(create_game):
    """Test that cards are correctly dealt from the deck"""
    game = create_game
    initial_deck_size = len(game.deck.deck)
    card = game.deck.deal_card()
    
    assert isinstance(card, tuple)  # Should return a tuple (num, suit)
    assert len(game.deck.deck) == initial_deck_size - 1  # Deck size should decrease

def test_deal_hole_cards(create_game):
    """Test that each player is dealt two hole cards."""
    game = create_game
    game.deal_hole_cards()
    
    for hand in game.hands:
        assert len(hand.get_cards()) == 4  # 2 cards (num, suit) for each player


def test_determine_winner_no_showdown(create_game):
    """Test determining the winner without a showdown (everyone else folded)."""
    game = create_game
    game.folded = [False, True, True]  # Only player 0 remains
    game.pot = 100
    
    game.determine_winner(showdown=False)
    assert game.players[0].balance == 200 + 100  # Winner gets the pot
    
def test_showdown_winner(create_random_game):
    """Test that the correct player wins during a showdown."""
    g = create_random_game
    g.folded = [False, False, False]  # No one folded
    g.hands[0].add_card('A', 's')
    g.hands[0].add_card('K', 's')
    g.hands[1].add_card('2', 'c')
    g.hands[1].add_card('7', 'h')
    g.hands[2].add_card('5', 'd')
    g.hands[2].add_card('9', 'c')
    g.community_cards.add_card('A', 'h')
    g.community_cards.add_card('K', 'd')
    g.community_cards.add_card('5', 's')
    g.community_cards.add_card('7', 's')
    g.community_cards.add_card('2', 's')

    g.pot = 300
    g.determine_winner(showdown=True)
    
    # Player 0 should win with two pairs (Aces and Kings)
    assert g.players[0].balance == 200 + 300  # Player 0 wins the pot
    

        
