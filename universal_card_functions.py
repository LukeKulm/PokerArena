def rank_to_num(rank):
    if rank == "T":
        return 10
    if rank == "J":
        return 11
    if rank == "Q":
        return 12
    if rank == "K":
        return 13
    if rank == "A":
        return 14
    else:
        return int(rank)


def suit_to_num(suit):
    if suit == "c":
        return 0
    if suit == "d":
        return 1
    if suit == "h":
        return 2
    else:  # spade
        return 3


def num_to_rank(num):
    if num == 10:
        return "T"
    if num == 11:
        return "J"
    if num == 12:
        return "Q"
    if num == 13:
        return "K"
    if num == 14:
        return "A"
    else:
        return str(num)


def num_to_suite(num):
    if num == 0:
        return "c"
    if num == 1:
        return "d"
    if num == 2:
        return "h"
    else:  # spade
        return "s"
    
def rank_to_prime(num):
    mapping = {2:2, 3:3, 4:5, 5:7, 6:11, 7:13, 8:17, 9:19, 10:23, 11:29, 12:31, 13:37, 14:41}
    return mapping[num]

def primify(num):
    mapping = {
        0b0001000000000000: 41, 
        0b0000100000000000: 37, 
        0b0000010000000000: 31, 
        0b0000001000000000: 29, 
        0b0000000100000000: 23,
        0b0000000010000000: 19, 
        0b0000000001000000: 17, 
        0b0000000000100000: 13,
        0b0000000000010000: 11, 
        0b0000000000001000: 7, 
        0b0000000000000100: 5, 
        0b0000000000000010: 3, 
        0b0000000000000001: 2
    }
    return mapping[num]
