import random
import score_hands
import itertools
TOTAL_GAMES_SIMULATED = 1000

suit_map_to_our_encoding = {0: "c", 1: "d", 2: "h", 3: "s"}
card_map_to_our_encoding = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
                            9: "9", 8: "8", 7: "7", 6: "6", 5: "5",
                            4: "4", 3: "3", 2: "2"}

cards = list(itertools.product(card_map_to_our_encoding.keys(),
             suit_map_to_our_encoding.keys()))


def convert_to_str_lst(lst):
    ret = []
    for item in lst:
        ret.append(card_map_to_our_encoding[item[0]])
        ret.append(suit_map_to_our_encoding[item[1]])
    return ret


def expected_win_rate(player_hand_cards, current_board, number_of_opps):
    games_won = 0

    hand_cards = []
    for i in range(len(player_hand_cards)//2):
        hand_cards.append((player_hand_cards[2*i], player_hand_cards[2*i+1]))
    board_cards = []
    for i in range(len(current_board)//2):
        if current_board[2*i] != 0:
            board_cards.append((current_board[2*i], current_board[i*2+1]))

    absent_board_cards = 5 - len(board_cards)
    already_dealt = hand_cards + board_cards

    remaining_deck = [card for card in cards if card not in already_dealt]

    for i in range(TOTAL_GAMES_SIMULATED):
        random.shuffle(remaining_deck)

        new_board_cards = remaining_deck[:absent_board_cards]

        other_player_cards = remaining_deck[absent_board_cards:
                                            absent_board_cards+number_of_opps*2]
        end_board = board_cards + new_board_cards

        this_player_all_cards = end_board + hand_cards
        _, score_metric = score_hands.best_hand_calc(
            convert_to_str_lst(this_player_all_cards))

        this_player_won = True

        for j in range(len(other_player_cards)//2):
            opp_player_hand_cards = other_player_cards[j*2:j*2+2]
            opp_player_all_cards = end_board + opp_player_hand_cards
            _, opp_score_metric = score_hands.best_hand_calc(
                convert_to_str_lst(opp_player_all_cards))
            if opp_score_metric >= score_metric:
                this_player_won = False
                break

        if this_player_won:
            games_won += 1
    win_rate = games_won / TOTAL_GAMES_SIMULATED

    return win_rate
