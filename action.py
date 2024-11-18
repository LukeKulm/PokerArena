from enum import Enum


class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    END_GAME = 4


class ActionInformation:
    def __init__(self, action, amount, all_in):
        self.action = action
        self.amount = amount
        self.is_player_all_in = all_in
