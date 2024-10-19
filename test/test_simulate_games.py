import simulate_games


def test_convert_to_str_lst():
    lst = [(9, 2), (14, 3)]
    res = simulate_games.convert_to_str_lst(lst)
    assert res == ['9', 'h', 'A', 's']


def test_royal_flush_results_in_max_win_rate():
    res = simulate_games.expected_win_rate(
        [14, 0, 13, 0], [12, 0, 11, 0, 10, 0], 6)
    assert res == 1.0


def test_royal_flush_in_community_cards_results_in_zero():
    res = simulate_games.expected_win_rate(
        [3, 3, 2, 2], [14, 0, 13, 0, 12, 0, 11, 0, 10, 0], 6)
    assert res == 0.0
