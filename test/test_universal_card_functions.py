import universal_card_functions


def test_rank_to_num():
    assert universal_card_functions.rank_to_num("T") == 10
    assert universal_card_functions.rank_to_num("J") == 11
    assert universal_card_functions.rank_to_num("Q") == 12
    assert universal_card_functions.rank_to_num("K") == 13
    assert universal_card_functions.rank_to_num("A") == 14
    for i in range(2, 10):
        assert universal_card_functions.rank_to_num(str(i)) == i
