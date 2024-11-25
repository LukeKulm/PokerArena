import pytest
import sys
import game
import os

from contextlib import redirect_stdout
from scripts.utils import get_not_busted


def test_all_ai_agents():
  players = [("MonteCarloQLearningHybrid", "saved_models/montecarlo_qlearning_hybrid.pth"), 
             ("QLearningAgent", "saved_models/q_network.pth"), 
             ("MonteCarlo", ""), 
             ("Random", ""),
             ("BCPlayer", ""),
             ("SmartBCPlayer", "")]
  i = 0
  games = 0

  with open(os.devnull, 'w') as fnull:
      with redirect_stdout(fnull):

          n = 100
          while i < n:
              g = game.Game(players, 200)
              games += 1
              while i < n and get_not_busted(g, 2) == len(players) and not g.user_ended:
                  g.step()
                  i += 1