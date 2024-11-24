import sys
import game
import os

from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from scripts.utils import get_not_busted


def main():
    """
    Simulates a game of Texas Hold'em
    """
    players = [("MonteCarloQLearningHybrid",
                "saved_models/q_network_with_montecarlo.pth"), ("QLearningAgent", "saved_models/q_network.pth")]
    i = 0
    balances = [[] for _ in players]
    sums = [0]*len(players)
    game_balances = [0]*len(players)
    games = 0

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):

            n = 100
            while i < n:
                g = game.Game(players, 200)
                games += 1
                while i < n and get_not_busted(g, 2) == len(players) and not g.user_ended:
                    print(f"We have simulated {i} hands")
                    g.step()
                    i += 1
                    for j in range(len(players)):
                        balances[j].append(
                            g.players[j].balance-(games*200)+sums[j])
                        game_balances[j] = g.players[j].balance
                for j in range(len(players)):
                    sums[j] += game_balances[j]

    colors = ["blue", "red", "green", "yellow", "black"]
    for j in range(len(players)):
        plt.plot(range(len(balances[j])), balances[j], color=colors[j])
    title = ""
    for j in range(len(players)):
        title += players[0][j]+" ("+colors[j]+"), "
    plt.title(f'{title}over {n} hands which took {games} games')
    plt.xlabel('# of hands')
    plt.ylabel('Net Gain')

    plt.savefig("my_plot.png")

    plt.show()


if __name__ == "__main__":
    main()
