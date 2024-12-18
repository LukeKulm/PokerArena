import pytest
from scripts.pluribus_extractor import *

def test_parse_community_cards():
  action = "d db 5c9s7c"
  parsed = parse_community_cards(action)
  assert parsed == ['5c', '9s', '7c']
  
def test_determine_stage():
  community_cards = Hand()
  assert determine_stage(community_cards) == 0
  community_cards.add_card('5', 'c')
  community_cards.add_card('9', 's')
  community_cards.add_card('7', 'c')
  assert determine_stage(community_cards) == 1
  community_cards.add_card('T', 'c')
  assert determine_stage(community_cards) == 2
  community_cards.add_card('5', 's')
  assert determine_stage(community_cards) == 3
  
def test_determine_dealer_position():
  actions = ['d dh p1 6d5s', 'd dh p2 Td2d', 'd dh p3 4dKh', 'd dh p4 7c4c', 'd dh p5 3s6c', 'd dh p6 7d8c', 'p3 f', 'p4 f', 'p5 f', 'p6 cbr 250', 'p1 f', 'p2 cc', 'd db 6sAsTh', 'p2 cc', 'p6 cbr 275', 'p2 cc', 'd db 7s', 'p2 cc', 'p6 cc', 'd db Ts', 'p2 cc', 'p6 cc', 'p2 sm Td2d', 'p6 sm']
  num_players = 6
  assert determine_dealer_position(actions, num_players) == 1 # could be 2, also dealer position never changes
  

  
  