import sys
import game


def main():
<<<<<<< HEAD
    g = game.Game(["Human", "Random"], 200)
=======
    g = game.Game(["Human", "Human"], 200)
>>>>>>> 11283f3ce30433575df9556d5bc27c118fb50305
    while get_not_busted(g) > 1:
        g.step()


def get_not_busted(g):
    num_players_not_busted = 0
    for player in g.players:
        if player.balance > 0:
            num_players_not_busted += 1
    return num_players_not_busted


if __name__ == "__main__":
    main()
