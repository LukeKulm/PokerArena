from collections import Counter
from itertools import combinations


class ScoreHandsError(Exception):
    pass


def hand_rank(hand):
    """
    Returns tuple with the hand's rank (category) and the tiebreak info.
    """
    ranks = sorted(["--23456789TJQKA".index(r) for r, _ in hand], reverse=True)
    rank_counts = Counter(ranks).most_common()
    is_flush = len(set(s for _, s in hand)) == 1
    is_straight = ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5

    if ranks == [14, 5, 4, 3, 2]:  # Account for A-5 straight
        is_straight = True
        ranks = [5, 4, 3, 2, 1]

    # Royal Flush
    if is_flush and ranks == [14, 13, 12, 11, 10]:
        return (9,)

    # Straight Flush
    if is_flush and is_straight:
        return (8, ranks[0])

    # Four of a Kind
    if rank_counts[0][1] == 4:
        return (7, rank_counts[0][0], rank_counts[1][0])

    # Full House
    if rank_counts[0][1] == 3 and rank_counts[1][1] == 2:
        return (6, rank_counts[0][0], rank_counts[1][0])

    # Flush
    if is_flush:
        return (5, ranks)

    # Straight
    if is_straight:
        return (4, ranks[0])

    # Three of a Kind
    if rank_counts[0][1] == 3:
        return (3, rank_counts[0][0], ranks)

    # Two Pair
    if rank_counts[0][1] == 2 and rank_counts[1][1] == 2:
        return (2, rank_counts[0][0], rank_counts[1][0], ranks)

    # One Pair
    if rank_counts[0][1] == 2:
        return (1, rank_counts[0][0], ranks)

    # High Card
    return (0, ranks)


def best_hand_calc(cards):
    """
    Takes a list of at least 5 cards and returns the best 5-card hand.
    """
    if len(cards) < 5:
        raise ScoreHandsError(
            "Cannot calculate best hand with less than 5 cards total.")
    if len(set(cards)) != len(cards):
        raise ScoreHandsError(
            f"Invalid hand. Duplicate cards exits.")

    # Calculate best 5 card hand
    best = max(combinations(cards, 5), key=hand_rank)
    return best, hand_rank(best)
