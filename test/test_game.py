import pytest
import game
import player


def test_game_step_random_players():
    """Test that game can run many times with random players"""
    for _ in range(1000):
        g = game.Game(
            [("Random", None), ("Random", None), ("Random", None)], 200)
        g.step()


def test_random_vs_montecarlo_players():
    """Test that game can run many times with random vs montecarlo players"""
    montecarlo_winnings = 0
    random_winnings = 0
    for _ in range(100):
        g = game.Game([("Random", None), ("MonteCarlo", None)], 200)
        g.step()
        if g.players[1].balance > g.players[0].balance:
            montecarlo_winnings += (g.players[1].balance - 200)
        elif g.players[0].balance > g.players[1].balance:
            random_winnings += (g.players[0].balance - 200)
    # this test is non-deterministic which is suspect
    assert montecarlo_winnings > random_winnings


@pytest.fixture
def create_game():
    """Fixture to create a game with 3 players."""
    players = [("Human", None), ("Random", None), ("Random", None)]
    return game.Game(players, start=200)


@pytest.fixture
def create_random_game():
    """Fixture to create a game with 3 random players."""
    players = [("Random", None), ("Random", None), ("Random", None)]
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
    assert len(game.deck.deck) == initial_deck_size - \
        1  # Deck size should decrease


def test_deal_hole_cards(create_game):
    """Test that each player is dealt two hole cards."""
    game = create_game
    game.deal_hole_cards()

    for hand in game.hands:
        # 2 cards (num, suit) for each player
        assert len(hand.get_cards()) == 4


def test_determine_winner_no_showdown(create_game):
    """Test determining the winner without a showdown (everyone else folded)."""
    game = create_game
    game.folded = [False, True, True]  # Only player 0 remains
    game.pot = 100
    game.over = False
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
    for player in g.players:
        player.in_hand_for = 100
    g.over = False
    g.determine_winner(showdown=True)

    # Player 0 should win with two pairs (Aces and Kings)
    assert g.players[0].balance == 200 + 300  # Player 0 wins the pot


def test_one_side_pot(create_random_game):
    game = create_random_game
    game.folder = [False, False, False]
    game.hands[0].add_card('A', 'c')
    game.hands[0].add_card('A', 'd')
    game.hands[1].add_card('K', 'c')
    game.hands[1].add_card('K', 's')
    game.hands[2].add_card('Q', 's')
    game.hands[2].add_card('Q', 'c')
    game.community_cards.add_card('A', 'h')
    game.community_cards.add_card('A', 's')
    game.community_cards.add_card('K', 'd')
    game.community_cards.add_card('K', 'h')
    game.community_cards.add_card('Q', 'd')

    for i in range(len(game.players)):
        if i == 0:
            game.players[i].in_hand_for = 100
        else:
            game.players[i].in_hand_for = 200
    game.over = False
    game.determine_winner(showdown=True)
    assert game.players[0].balance == 500
    assert game.players[1].balance == 400
    assert game.players[2].balance == 200


def test_multiple_side_pots(create_random_game):
    game = create_random_game
    game.folder = [False, False, False]
    game.hands[0].add_card('A', 'c')
    game.hands[0].add_card('A', 'd')
    game.hands[1].add_card('K', 'c')
    game.hands[1].add_card('K', 's')
    game.hands[2].add_card('Q', 's')
    game.hands[2].add_card('Q', 'c')
    game.community_cards.add_card('A', 'h')
    game.community_cards.add_card('A', 's')
    game.community_cards.add_card('K', 'd')
    game.community_cards.add_card('K', 'h')
    game.community_cards.add_card('Q', 'd')

    game.players[0].in_hand_for = 100
    game.players[1].in_hand_for = 200
    game.players[2].in_hand_for = 300
    game.over = False
    game.determine_winner(showdown=True)
    assert game.players[0].balance == 500
    assert game.players[1].balance == 400
    assert game.players[2].balance == 300


def test_tie_in_side_pot(create_random_game):
    game = create_random_game
    game.folder = [False, False, False]
    game.hands[0].add_card('A', 'c')
    game.hands[0].add_card('A', 'd')
    game.hands[1].add_card('K', 'c')
    game.hands[1].add_card('K', 's')
    game.hands[2].add_card('K', 'd')
    game.hands[2].add_card('K', 'H')
    game.community_cards.add_card('A', 'h')
    game.community_cards.add_card('A', 's')
    game.community_cards.add_card('Q', 's')
    game.community_cards.add_card('Q', 'c')
    game.community_cards.add_card('Q', 'd')

    game.players[0].in_hand_for = 100
    game.players[1].in_hand_for = 200
    game.players[2].in_hand_for = 200
    game.over = False
    game.determine_winner(showdown=True)
    assert game.players[0].balance == 500
    assert game.players[1].balance == 300
    assert game.players[2].balance == 300


def test_tie_in_main_pot(create_random_game):
    game = create_random_game
    game.folder = [False, False, False]
    game.hands[0].add_card('2', 'c')
    game.hands[0].add_card('6', 'd')
    game.hands[1].add_card('K', 'c')
    game.hands[1].add_card('K', 's')
    game.hands[2].add_card('K', 'd')
    game.hands[2].add_card('K', 'H')
    game.community_cards.add_card('A', 'h')
    game.community_cards.add_card('A', 's')
    game.community_cards.add_card('Q', 's')
    game.community_cards.add_card('Q', 'c')
    game.community_cards.add_card('8', 'd')

    game.players[0].in_hand_for = 200
    game.players[1].in_hand_for = 200
    game.players[2].in_hand_for = 200
    game.over = False
    game.determine_winner(showdown=True)
    assert game.players[0].balance == 200
    assert game.players[1].balance == 500
    assert game.players[2].balance == 500
