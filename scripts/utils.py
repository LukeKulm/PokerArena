def get_not_busted(g, n):
    """
    Returns the number of players with a more than n chips

    param g: the game.Game() object
    """
    num_players_not_busted = 0
    for player in g.players:
        if player.balance > n:
            num_players_not_busted += 1
    return num_players_not_busted
