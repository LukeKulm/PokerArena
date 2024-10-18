import csv
import pytest
import score_hands

TEST_DATA_PATH = "./data/poker_hands/poker-hand-testing.data"

suit_map_to_our_encoding = {"1": "c", "2": "d", "3": "h", "4": "s"}
card_map_to_our_encoding = {"1": "A", "13": "K", "12": "Q", "11": "J", "10": "T",
                            "9": "9", "8": "8", "7": "7", "6": "6", "5": "5",
                            "4": "4", "3": "3", "2": "2"}


def convert_to_correct_format(hand):
    cards = []

    for i in range((len(hand)-1)//2):
        first = 2 * i
        second = 2 * i + 1
        suit = suit_map_to_our_encoding[hand[first]]
        number = card_map_to_our_encoding[hand[second]]
        cards.append((number, suit))

    return cards


class TestBestHandCalc:
    def test_calculates_properly(self):
        with open(TEST_DATA_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]

        for hand in data:
            res = score_hands.best_hand_calc(convert_to_correct_format(hand))
            assert (res[1][0]) == int(hand[-1])

    def test_too_few_cards(self):
        hands = []
        with pytest.raises(score_hands.ScoreHandsError, match="Cannot calculate best hand with less than 5 cards total."):
            score_hands.best_hand_calc(hands)

    def test_same_card_included_twice(self):
        hands = [("2", "c"), ("2", "c"), ("3", "c"), ("4", "c"),
                 ("5", "c"), ("6", "c"), ("7", "c"), ]
        with pytest.raises(score_hands.ScoreHandsError, match="Invalid hand. Duplicate cards exits."):
            score_hands.best_hand_calc(hands)
