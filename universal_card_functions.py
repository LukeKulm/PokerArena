def rank_to_num(self, rank):
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


def suit_to_num(self, suit):
    if suit == "c":
        return 0
    if suit == "d":
        return 1
    if suit == "h":
        return 2
    else:  # spade
        return 3


def num_to_rank(self, num):
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

def num_to_suite(self, num):
    if num == 0:
        return "c"
    if num == 1:
        return "d"
    if num == 2:
        return "h"
    else:  # spade
        return "s"
