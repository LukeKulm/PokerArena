import sys
import game
import torch
import os
import numpy as np

    
def aggregate():
    """
    Simulates a game of Texas Hold'em
    """
    g = game.Game(["DataAggregator", "DataAggregator"], 200)
    while get_not_busted(g) > 1 and not g.user_ended:
        g.step()
    print("The final stacks are: "+str(g.players[0].balance)+" "+str(g.players[1].balance))
    print("There was "+ str(g.pot)+" in the pot.")
    x1 = g.players[0].x
    y1 = g.players[0].y
    x2 = g.players[1].x
    y2 = g.players[1].y
    new_data = torch.tensor(np.array(x1+x2), dtype=torch.float32)
    new_labels = torch.tensor(np.array(y1+y2), dtype=torch.float32)
    if not os.path.exists('data/expert_policy.pt'):
        torch.save((new_data, new_labels), 'data/expert_policy.pt')
    else:
        existing_data, existing_labels = torch.load('data/expert_policy.pt', weights_only=True)
        updated_data = torch.cat((existing_data, new_data), dim=0)
        updated_labels = torch.cat((existing_labels, new_labels), dim=0)
        torch.save((updated_data, updated_labels), 'data/expert_policy.pt')



def get_not_busted(g):
    """
    Returns the number of players with a nonzero stack

    param g: the Game() object
    """
    num_players_not_busted = 0
    for player in g.players:
        if player.balance > 0:
            num_players_not_busted += 1
    return num_players_not_busted


if __name__ == "__main__":
    aggregate()
