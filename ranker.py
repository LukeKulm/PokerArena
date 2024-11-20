import numpy as np
from parse_hands import Parser

class Ranker():
    """
    Class that provides ranking of hands according to equivalence classes
    """

    def __init__(self, parser, hand):
        self.parser = parser
        self.parser.parse()
        self.data = self.parser.table()
        self.hand = hand

    def pre_flop(self):
        pass