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
  return 0
  
def test_encode_hand_history():
  actions = ['d dh p1 3c9s', 'd dh p2 6d5s', 'd dh p3 9dTs', 'd dh p4 2sQs', 'd dh p5 AdKd', 'd dh p6 7cTc', 'p3 f', 'p4 f', 'p5 cbr 225', 'p6 f', 'p1 f', 'p2 f']
  players = ['MrWhite', 'Gogo', 'Budd', 'Eddie', 'Bill', 'Pluribus']
  finishing_stacks = [9950, 10250, 10000, 10000, 10000, 9800]
  min_bet = 0 #doesnt matter
  blinds_or_straddles = [50, 100, 0, 0, 0, 0]
  antes = [0, 0, 0, 0, 0, 0]
  
  player_index = 0
  encoding = PlayerActionGameState(
    num_players=6,
    num_players_folded=3,
    player_index=1,
    player_cards='3c9s',
    player_balance=9950,
    dealer_position=1, #or 2 for player 2
    stage=0,
    pot=150+225,
    community_cards=Hand(),
    curr_bet=225,
    amount_in_for=50, #maybe this is 0
  ).encode()
  
  assert encoding == encode_hand_history(antes, blinds_or_straddles, min_bet, actions, players, finishing_stacks)[player_index][0]
  