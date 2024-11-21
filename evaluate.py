import sys
import game
import os

from contextlib import redirect_stdout
import matplotlib.pyplot as plt


def main():
    """
    Simulates a game of Texas Hold'em
    """
    players = ["Random", "SmartBCPlayer"]
    i = 0
    balances = [[] for _ in players]
    sums = [0]*len(players)
    game_balances = [0]*len(players)
    games = 0

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            
            n = 1000
            while i<n:
                g = game.Game(players, 200)
                games+=1
                while i<n and  get_not_busted(g) == len(players) and not g.user_ended:
                    print(f"We have simulated {i} hands")
                    g.step()
                    i+=1
                    for j in range(len(players)):
                        balances[j].append(g.players[j].balance-(games*200)+sums[j])
                        game_balances[j] = g.players[j].balance
                for j in range(len(players)):
                    sums[j]+=game_balances[j] 
                
    colors = ["blue", "red", "green", "yellow", "black"]
    for j in range(len(players)):
        plt.plot(range(len(balances[j])), balances[j], color=colors[j])
    title = ""
    for j in range(len(players)):
        title+=players[j]+" ("+colors[j]+"), "
    plt.title(f'{title}over {n} hands which took {games} games')            
    plt.xlabel('# of hands')            
    plt.ylabel('Net Gain')            

    plt.savefig("my_plot.png")

    plt.show()
    

def get_not_busted(g):
    """
    Returns the number of players with a nonzero stack

    param g: the Game() object
    """
    num_players_not_busted = 0
    for player in g.players:
        if player.balance >2:
            num_players_not_busted += 1
    return num_players_not_busted


if __name__ == "__main__":
    main()
