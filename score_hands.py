from collections import Counter
from itertools import combinations

# Define the ranks and suits
RANKS = "23456789TJQKA"
SUITS = "cdhs"

def hand_rank(hand):
    """
    Returns a tuple with the hand's rank (category) and the tiebreaker information.
    E.g., for a full house, it returns (6, rank of three of a kind, rank of pair).
    """
    ranks = sorted(["--23456789TJQKA".index(r) for r, s in hand], reverse=True)
    rank_counts = Counter(ranks).most_common()
    is_flush = len(set(s for r, s in hand)) == 1
    is_straight = ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5
    
    if ranks == [14, 5, 4, 3, 2]:  # Adjust for A-5 straight
        is_straight = True
        ranks = [5, 4, 3, 2, 1]

    # Royal Flush
    if is_flush and ranks == [14, 13, 12, 11, 10]:
        return (10,)
    
    # Straight Flush
    if is_flush and is_straight:
        return (9, ranks[0])
    
    # Four of a Kind
    if rank_counts[0][1] == 4:
        return (8, rank_counts[0][0], rank_counts[1][0]) 
    
    # Full House
    if rank_counts[0][1] == 3 and rank_counts[1][1] == 2:
        return (7, rank_counts[0][0], rank_counts[1][0])
    
    # Flush
    if is_flush:
        return (6, ranks)
    
    # Straight
    if is_straight:
        return (5, ranks[0])
    
    # Three of a Kind
    if rank_counts[0][1] == 3:
        return (4, rank_counts[0][0], ranks)
    
    # Two Pair
    if rank_counts[0][1] == 2 and rank_counts[1][1] == 2:
        return (3, rank_counts[0][0], rank_counts[1][0], ranks)
    
    # One Pair
    if rank_counts[0][1] == 2:
        return (2, rank_counts[0][0], ranks)
    
    # High Card
    return (1, ranks)


def best_hand_calc(cards):
    """
    Takes a list of 7 cards (5 community + 2 from player's hand) and returns the best 5-card hand.
    """
    # Generate all 5-card combinations
    best = max(combinations(cards, 5), key=hand_rank)
    return best, hand_rank(best)