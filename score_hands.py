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
        return (10,)  # Highest rank
    
    # Straight Flush
    if is_flush and is_straight:
        return (9, ranks[0])  # Straight flush, tiebreaker is the top card
    
    # Four of a Kind
    if rank_counts[0][1] == 4:
        return (8, rank_counts[0][0], rank_counts[1][0])  # Four of a kind, tiebreaker
    
    # Full House
    if rank_counts[0][1] == 3 and rank_counts[1][1] == 2:
        return (7, rank_counts[0][0], rank_counts[1][0])  # Full house
    
    # Flush
    if is_flush:
        return (6, ranks)  # Flush
    
    # Straight
    if is_straight:
        return (5, ranks[0])  # Straight
    
    # Three of a Kind
    if rank_counts[0][1] == 3:
        return (4, rank_counts[0][0], ranks)  # Three of a kind
    
    # Two Pair
    if rank_counts[0][1] == 2 and rank_counts[1][1] == 2:
        return (3, rank_counts[0][0], rank_counts[1][0], ranks)  # Two pair
    
    # One Pair
    if rank_counts[0][1] == 2:
        return (2, rank_counts[0][0], ranks)  # One pair
    
    # High Card
    return (1, ranks)  # High card


def best_hand(cards):
    """
    Takes a list of 7 cards (5 community + 2 from player's hand) and returns the best 5-card hand.
    """
    # Generate all 5-card combinations
    best = max(combinations(cards, 5), key=hand_rank)
    return best, hand_rank(best)


def compare_hands(hand1, hand2, community_cards):
    """
    Compares two hands (each with 2 cards) plus the community cards to determine the winner.
    """
    full_hand1 = hand1.get_cards() + community_cards.get_cards()
    full_hand2 = hand2.get_cards() + community_cards.get_cards()

    _, score1 = best_hand(full_hand1)
    _, score2 = best_hand(full_hand2)
    
    if score1 > score2:
        return "Hand 1 wins"
    elif score2 > score1:
        return "Hand 2 wins"
    else:
        return "It's a tie!"