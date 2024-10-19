import pytest
import game


class TestGameSteps:
    def test_game_step_random_players(self):
      #make sure game doesnt crash when run 100 times
      g = game.Game(["Random", "Random"], 200)
      for i in range(100):
        g.step()
      assert game == game
      
    def test_everyone_folds_last_wins(self):
      g = game.Game([["Random", "Random", "Random"], 200])
      g.deal_hole_cards()
      g.stage = 1
      #TODO: finish this
      assert game == game
        
