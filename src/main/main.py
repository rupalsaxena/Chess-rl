import os

from agents.Chess_QLearning import Chess_QLearning
from agents.Chess_Random import Chess_Random
from agents.Chess_SARSA import Chess_SARSA
from agents.Chess_Analyse import Analyse_Env
from utils.helpers import helpers


CONFIG_PATH = "src/main/configs/sarsa.yaml"
#CONFIG_PATH = "src/main/configs/q-learning.yaml"
#CONFIG_PATH = "src/main/configs/random.yaml"
#CONFIG_PATH = "src/main/configs/sarsa_explode_gradients.yaml"

CONFIG_PATH = os.environ["CONFIG_PATH"]

def main():
    h = helpers()
    config = h.read_yaml(CONFIG_PATH)
    algo = config["algo"]

    if config["analyse"]:
        analyse_chess = Analyse_Env()
        analyse_chess.analyse()

    if algo == "sarsa":
        rl_algo = Chess_SARSA(config)
    elif algo == "q-learning":
        rl_algo = Chess_QLearning(config)
    else:
        rl_algo = Chess_Random()

    rl_algo.train()
    rl_algo.plot()


if __name__ == "__main__":
    main()