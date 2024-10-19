import simulate
import game


def test_get_not_busted_none_busted():
    g = game.Game(["Human", "Random"], 200)
    assert simulate.get_not_busted(g) == 2


def test_get_not_busted_one_busted():
    g = game.Game(["Human", "Random"], 200)
    g.players[0].balance = 0
    assert simulate.get_not_busted(g) == 1


def test_get_not_busted_all_busted():
    g = game.Game(["Human", "Random"], 200)
    g.players[0].balance = 0
    g.players[1].balance = 0
    assert simulate.get_not_busted(g) == 0
