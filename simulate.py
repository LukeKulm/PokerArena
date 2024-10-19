import sys
import game

def main():
    g = game.Game(["Human", "Random"], 200)
    g.step()

if __name__ == "__main__":
    main()