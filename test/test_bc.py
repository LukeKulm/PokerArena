import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
sys.path.insert(0, "scripts")

import pytest 
import game
import torch
from evaluate import get_not_busted
import numpy as np
import os
from train_bc import training_fn
from player import BCPlayer, round_prediction

# uses random players to generate a set of 1000 random game states, labels them all with a fold, then trains a model on this policy and verifies that it always folds
def test_always_calls():
    players = ([("Random", None), ("Random", None)])
    n = 1000
    i=0
    while i<n:
        g = game.Game(players, 200)
        while i<n and  get_not_busted(g, 2) == len(players) and not g.user_ended:
            print(f"We have simulated {i} hands")
            g.step()
            i = len(g.players[0].x) + len(g.players[1].x)
    x1 = g.players[0].x
    x2 = g.players[1].x
    if not os.path.exists('data/call_policy.pt'):
        
        y = [[1, 0, 0]]*len(x1+x2)
        new_data = torch.tensor(np.array(x1+x2), dtype=torch.float32)
        new_labels = torch.tensor(np.array(y), dtype=torch.float32)
        assert len(new_data) == len(new_labels)

        torch.save((new_data, new_labels), 'data/call_policy.pt')

    
    # train the model
    if not os.path.exists('calling_checkpoint.pth'):
        training_fn('data/call_policy.pt', 'calling_checkpoint.pth')

    caller = BCPlayer(200, 1, model_name='calling_checkpoint.pth')
    errors = 0
    for state in (x1+x2):
        state_tensor = torch.from_numpy(state).float()
        prediction = caller.model.forward(state_tensor)
        move = prediction[0]
        assert 1 == round_prediction(move)

def test_always_folds():
    players = ([("Random", None), ("Random", None)])
    n = 1000
    i=0
    while i<n:
        g = game.Game(players, 200)
        while i<n and  get_not_busted(g, 2) == len(players) and not g.user_ended:
            print(f"We have simulated {i} hands")
            g.step()
            i = len(g.players[0].x) + len(g.players[1].x)
    x1 = g.players[0].x
    x2 = g.players[1].x
    y = [[2, 0, 0]]*len(x1+x2)
    if not os.path.exists('data/fold_policy.pt'):
        y = [[0, 0, 0]]*len(x1+x2)
        new_data = torch.tensor(np.array(x1+x2), dtype=torch.float32)
        new_labels = torch.tensor(np.array(y), dtype=torch.float32)
        assert len(new_data) == len(new_labels)

        torch.save((new_data, new_labels), 'data/fold_policy.pt')

    if not os.path.exists('folding_checkpoint.pth'):
        # train the model
        training_fn('data/fold_policy.pt', 'folding_checkpoint.pth')

    caller = BCPlayer(200, 1, model_name='folding_checkpoint.pth')
    errors = 0
    for state in (x1+x2):
        state_tensor = torch.from_numpy(state).float()
        prediction = caller.model.forward(state_tensor)
        move = prediction[0]
        assert 0 == round_prediction(move)
        
def test_always_raises():
    players = ([("Random", None), ("Random", None)])
    n = 1000
    i=0
    while i<n:
        g = game.Game(players, 200)
        while i<n and  get_not_busted(g, 2) == len(players) and not g.user_ended:
            print(f"We have simulated {i} hands")
            g.step()
            i = len(g.players[0].x) + len(g.players[1].x)
    x1 = g.players[0].x
    x2 = g.players[1].x
    
    if not os.path.exists('data/raise_policy.pt'):
        y = [[2, 0, 0]]*len(x1+x2)
        new_data = torch.tensor(np.array(x1+x2), dtype=torch.float32)
        new_labels = torch.tensor(np.array(y), dtype=torch.float32)
        assert len(new_data) == len(new_labels)

        torch.save((new_data, new_labels), 'data/raise_policy.pt')

    
    # train the model
    if not os.path.exists('raising_checkpoint.pth'): 
        training_fn('data/raise_policy.pt', 'raising_checkpoint.pth')

    caller = BCPlayer(200, 1, model_name='raising_checkpoint.pth')
    errors = 0
    for state in (x1+x2):
        state_tensor = torch.from_numpy(state).float()
        prediction = caller.model.forward(state_tensor)
        move = prediction[0]
        assert 2 == round_prediction(move)


