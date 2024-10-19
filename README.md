# JK_LCMJ_ctb93_jcs547_lbk73_mmw243
This is a repo to train an AI agent to play poker \
Currently our state for a game does not represent all players betting history. This is done to 
simplify the state representation and there is still strategy involved even if you do not know
what each specific player bet. We can update this in the future if we want a more advanced AI.

TODO for submission today:
    
    find out where hand object is being printed to console and remove it
    
    unit testing
    writeup
        readme
        comments/structuring
        demo plan

Script:

Cole will demonstrate the following:
1. Demonstrate the training and testing of the hand strength predictor model (run python hand_predictor.py and see printed info in stdout).
2. Demonstrate the playing of a MonteCarlo agent against Randoms via adding “MonteCarlo” to the list of game.Game initialization in simulate.py. Run python simulate.py.
