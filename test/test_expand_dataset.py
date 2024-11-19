import pytest
import torch
import expand_dataset

def test_swap():
    pre = torch.zeros(4)
    pre[1] = 1
    pre[2] = 2
    pre[3] = 3
    post = expand_dataset.swap(pre, 0, 2, 1, 3)
    print(post)
    assert post[0] == 2 and post[1] == 3 and post[2] == 0 and post[3] == 1
    assert pre[0] == 0 and pre[3] == 3
    

def test_preflop_perms():
    state = torch.zeros(23)
    state[10] = 1
    state[11] = 2
    state[12] = 3
    state[13] = 4
    perms = expand_dataset.get_perms(state)

    assert len(perms) == 2
    assert perms[1][10] == 3 and perms[1][12] == 1
    assert perms[1][11] == 4 and perms[1][13]== 2
def test_flop():
    state = torch.zeros(23)
    state[7] = 1
    perms = expand_dataset.get_perms(state)
    assert len(perms) == 12
def test_turn():
    state = torch.zeros(23)
    state[7] = 2
    perms = expand_dataset.get_perms(state)
    assert len(perms) == 48
def test_river():
    state = torch.zeros(23)
    state[7] = 3
    perms = expand_dataset.get_perms(state)
    assert len(perms) == 240