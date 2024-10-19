import sys
import game


def main():
<<<<<<< HEAD
    g = game.Game(["Human", "Human", "Human"], 200)
=======
    g = game.Game(["Human", "Random"], 200)
>>>>>>> d86a2aa42961f1bafe7763d81bfeb78becc806f6
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
