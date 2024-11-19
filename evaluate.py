import sys
import game
import os

from contextlib import redirect_stdout
import matplotlib.pyplot as plt

def main():
    """
    Simulates a game of Texas Hold'em
    """
    
    i = 0
    mc_balance = [0]
    random_balance = [0]
    mc_in = 0
    rand_in = 0
    mc_sum = 0
    rand_sum = 0

    # with open(os.devnull, 'w') as fnull:
    #     with redirect_stdout(fnull):
    while i<50:
        g = game.Game(["BCPlayer", "Random"], 200)
        mc_in+=1
        rand_in+=1
        while i<50 and  get_not_busted(g) > 1 and not g.user_ended:
            print(f"We have simulated {i} hands")
            g.step()
            i+=1
            mc_balance.append(g.players[0].balance-(mc_in*200)+mc_sum)
            random_balance.append(g.players[1].balance-(rand_in*200)+rand_sum)
            mc_game = g.players[0].balance
            rand_game = g.players[1].balance
        mc_sum+=mc_game
        rand_sum+=rand_game
    print(f"BC folded {g.players[0].folds} hands")
    print(mc_balance[-1])
    print(random_balance[-1])
    plt.plot(range(len(mc_balance)), mc_balance, color='blue')
    plt.plot(range(len(random_balance)), random_balance, color='red')
    plt.title('Monte Carlo vs Random')            
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
        if player.balance > 0:
            num_players_not_busted += 1
    return num_players_not_busted


if __name__ == "__main__":
    main()
