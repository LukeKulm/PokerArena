import sys
import game
import torch
import os
import numpy as np

def sim():
    """
    Simulates a game of Texas Hold'em
    """

    g = game.Game(["BCPlayer", "Human"], 200)
    while get_not_busted(g) > 1 and not g.user_ended:
        g.step()
    

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
    sim()
