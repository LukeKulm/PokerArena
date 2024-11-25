import universal_card_functions


def test_rank_to_num():
    assert universal_card_functions.rank_to_num("T") == 10
    assert universal_card_functions.rank_to_num("J") == 11
    assert universal_card_functions.rank_to_num("Q") == 12
    assert universal_card_functions.rank_to_num("K") == 13
    assert universal_card_functions.rank_to_num("A") == 14
    for i in range(2, 10):
        assert universal_card_functions.rank_to_num(str(i)) == i


def test_suit_to_num():
    assert universal_card_functions.suit_to_num("c") == 0
    assert universal_card_functions.suit_to_num("d") == 1
    assert universal_card_functions.suit_to_num("h") == 2
    assert universal_card_functions.suit_to_num("s") == 3


def test_num_to_rank():
    assert universal_card_functions.num_to_rank(10) == "T"
    assert universal_card_functions.num_to_rank(11) == "J"
    assert universal_card_functions.num_to_rank(12) == "Q"
    assert universal_card_functions.num_to_rank(13) == "K"
    assert universal_card_functions.num_to_rank(14) == "A"
    for i in range(2, 10):
        assert universal_card_functions.num_to_rank(i) == str(i)


def test_num_to_suite():
    assert universal_card_functions.num_to_suite(0) == "c"
    assert universal_card_functions.num_to_suite(1) == "d"
    assert universal_card_functions.num_to_suite(2) == "h"
    assert universal_card_functions.num_to_suite(3) == "s"
