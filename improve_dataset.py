import torch
import score_hands

"""
Desciption of new state:
0 number of players
1 position
2 dealer position
3 stage of hand
4 pot
5 stack
6 bet
7 high card
8 second high card
9 pocket pair, 0 if not
10 pockets suited, 0 if not

10 highest pair, 0 if none
11 second highest pair, 0 if none
12 highest trips, 0 if none
13 quads, 0 if none
14 flush draw
15 flush
16 open ended straight draw
17 straight
"""

def main():
    state_dict = torch.load("data/expanded_expert_data.pt")
    data, labels = state_dict
    better_data = torch.empty((0,) +(11,))
    better_labels = torch.empty((0,) + (3,))

    for old, pred in zip(data, labels):
        
        better_data = torch.cat((better_data, make_new_state(old).unsqueeze(0)), dim=0)
        better_labels = torch.cat((better_labels, pred.unsqueeze(0)), dim=0)
    
    torch.save((better_data, better_labels), 'data/improved_expert_data.pt')

def make_new_state(old):
    state = torch.empty(11)
    state[0] = old[0]
    state[1] = old[1]
    state[2] = old[6]
    state[3] = old[7]
    state[4] = old[9]
    state[5] = old[20]
    state[6] = old[21]-old[22]
    state[7] = max(old[10], old[12])
    state[8] = min(old[10], old[12])
    if state[7] == state[8]:
        state[9] = state[8]
    else:
        state[9] = 0
    if old[11] == old[13]:
        state[10] = 1
    else:
        state[10] = 0

    return state

if __name__ == "__main__":
    main()