import sys
import argparse
import game
import torch
import os
import numpy as np
from player import PLAYER_TYPES
from scripts.utils import get_not_busted


class SimulateError(Exception):
    pass


def normal_sim():
    """
    Simulates a game of Texas Hold'em
    """
    counts = {}
    for p_type in PLAYER_TYPES:
        num_of_p_type = int(input(f"Enter the number of {p_type} players: "))
        if num_of_p_type > 0:
            counts[p_type] = num_of_p_type

    lst = []
    for p_type in counts.keys():
        count = counts[p_type]
        for _ in range(count):
            lst.append(p_type)
    print(lst)
    if len(lst) > 10:
        raise SimulateError(
            f"Too many players: {len(lst)}. Max allowed is 10.")
    if len(lst) < 2:
        raise SimulateError(f"Too few players: {len(lst)}. Min allowed is 2.")

    g = game.Game(lst, 200)

    while get_not_busted(g, 0) > 1 and not g.user_ended:
        g.step()


def aggregate():
    """
    Simulates a game of Texas Hold'em
    """
    g = game.Game(["DataAggregator", "DataAggregator"], 200)
    while get_not_busted(g, 0) > 1 and not g.user_ended:
        g.step()
    print("The final stacks are: " +
          str(g.players[0].balance)+" "+str(g.players[1].balance))
    print("There was " + str(g.pot)+" in the pot.")
    x1 = g.players[0].x
    y1 = g.players[0].y
    x2 = g.players[1].x
    y2 = g.players[1].y
    new_data = torch.tensor(np.array(x1+x2), dtype=torch.float32)
    new_labels = torch.tensor(np.array(y1+y2), dtype=torch.float32)
    if not os.path.exists('data/expert_policy.pt'):
        torch.save((new_data, new_labels), 'data/expert_policy.pt')
    else:
        existing_data, existing_labels = torch.load(
            'data/expert_policy.pt', weights_only=True)
        updated_data = torch.cat((existing_data, new_data), dim=0)
        updated_labels = torch.cat((existing_labels, new_labels), dim=0)
        torch.save((updated_data, updated_labels), 'data/expert_policy.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poker Simulation Argument Parser")
    parser.add_argument("--aggregation_game", action="store_true",
                        help="Aggregation game activation")
    args = parser.parse_args()
    if args.aggregation_game:
        aggregate()
    else:
        normal_sim()
