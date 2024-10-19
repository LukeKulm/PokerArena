import sys
import game


def main():
    g = game.Game(["Human", "Human"], 200)
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
