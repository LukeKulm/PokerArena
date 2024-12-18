import sys
import game
import os

from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from scripts.utils import get_not_busted
import torch
import argparse
import player as p


def main(save_model, save_path):
    """
    Simulates a game of Texas Hold'em. This is a slightly adapted version of evaluate.py.
    """
    start_balances = 200
    players = [("MonteCarloQLearningHybrid", None), ("MonteCarlo", None), ("Random", None),
                ("QLearningAgent", "saved_models/q_network.pth"), ("MonteCarlo", None)]
    i = 0
    balances = [[] for _ in players]
    sums = [0]*len(players)
    game_balances = [0]*len(players)
    games = 0
    n = 1000
    g = game.Game(players, start_balances)
    while i < n:
        for player in g.players:
            player.train = True
            player.balance = start_balances
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

    if save_model:
        q_learning_player = g.players[0]
        torch.save(q_learning_player.q_network.state_dict(), save_path)
        
    colors = ["blue", "red", "green", "yellow", "black"]
    for j in range(len(players)):
        plt.plot(range(len(balances[j])), balances[j], color=colors[j])
    title = ""
    for j in range(len(players)):
        title += players[j][0]+" ("+colors[j]+"), "
    plt.title(f'{title}over {n} hands which took {games} games')
    plt.xlabel('# of hands')
    plt.ylabel('Net Gain')

    plt.savefig("my_plot.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Pytorch Model Parser")
    parser.add_argument("--save_model", action="store_true",
                        help="If active, saves the model to specified location")
    parser.add_argument("--save_path", type=str, default="q_network.pth",
                        help="Path to save the model. Defaults to q_network.pth")
    args = parser.parse_args()

    main(args.save_model, args.save_path)
