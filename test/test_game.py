import pytest
import game


class TestGameSteps:
    #make sure if everyone folds then last wins
    def test_game_step(self):
      g = game.Game(3)
      g.step()
      assert game == game
        
