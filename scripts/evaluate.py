import sys
import game
import os

from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from scripts.utils import get_not_busted
import argparse
import player


def main(advanced_tracking: bool):
    """
    Simulates a game of Texas Hold'em
    """
    players = [("QLearningAgent",
                "saved_models/q_learning_agent.pth"),
                ("MonteCarloQLearningHybrid", "saved_models/montecarlo_q_hybrid.pth"), 
                ("PokerTheoryQAgent",
                "saved_models/poker_theory_model.pth"),
                ("Random", None),
                ("Random", None)]
    
    i = 0
    balances = [[] for _ in players]
    sums = [0]*len(players)
    game_balances = [0]*len(players)
    games = 0
    if advanced_tracking:
        wrappers = [player.ActionTracker() for _ in players]

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):

            n = 100
            while i < n:
                g = game.Game(players, 200)

                if advanced_tracking:
                    for wrapper, new_player in zip(wrappers, g.players):
                        wrapper.underlying_agent = new_player
                    g.players = wrappers

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

    if advanced_tracking:
        for i in range(len(wrappers)):
            print(f"____________player{i} statistics ({players[i][0]})_____________")
            wrappers[i].print_compiled_actions()



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
    parser.add_argument("--advanced_tracking", action="store_true",
                        help="If active, saves the model to specified location")
    args = parser.parse_args()
    main(args.advanced_tracking)
