# JK_LCMJ_ctb93_jcs547_lbk73_mmw243
This project, Poker AI Showdown, is our attempt at training AI Agents to play Texas Hold'em poker

Final Demo Script:

James will demonstrate the following:
1. Demonstrate the poker gym environment that we used to train and evaluate the models, 
using the command: python -m scripts.simulate from the main directory, with one human player
and one PokerTheoryQAgent (after the prompt, simply pressing enter uses the pre-loaded model) to make the demo easier to follow

Max will demonstrate the following:
1. Demonstrate the poker game calculator, which wraps the AI agents that we've made so far
into a GUI that allows players to input a game state and receive recommendations. Max will run the
command: streamlit run calculator.py

Luke, having just had wisdom tooth surgery, will not be presenting.

Cole will demonstrate the following:
1. The evaluation script, which allows us to compare the performance of different AI
poker bots against each non-AI agents, but also against each other. It will be run
for 10 iterations, with a QLearningAgent, a MonteCarloQLearningHybrid, a PokerTheoryQAgent (all AI bots),
and a MonteCarlo and Random bot (not AI). The command is: python -m scripts.evaluate.
