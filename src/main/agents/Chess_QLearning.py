import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from env.Chess_env import Chess_Env
from env.generate_game import generate_game
from utils.NN import NN
from utils.helpers import helpers
import matplotlib.pyplot as plt


class Chess_QLearning:
    def __init__(self, config) -> None:
        self.N_h = config["N_h"]
        self.N_episodes = config["N_episodes"]
        self.epsilon_0 = config["epsilon_0"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.optimizer = config["optimizer"]
        self.momentum = config["momentum"]

        self.nn = NN()
        self.h = helpers()

        # initialize 4x4 chess board env
        self.env = Chess_Env(4)

    def train(self):
        print('Training q-learning and initializing seed')
        np.random.seed(3)
        S, X, allowed_a = self.env.Initialise_game()
        N_a = np.shape(allowed_a)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS

        N_in = np.shape(X)[0]  ## INPUT SIZE

        ## INITALISE YOUR NEURAL NETWORK...
        ##Xavier init
        W1 = np.random.randn(N_in, self.N_h) * np.sqrt(1 / (N_in))
        W2 = np.random.randn(self.N_h, N_a) * np.sqrt(1 / self.N_h)

        b1 = np.zeros([self.N_h])
        b2 = np.zeros([N_a])

        if self.optimizer == "rmsprop":
            sdw1 = np.ones((N_in, self.N_h))
            sdw2 = np.ones([self.N_h])
            sdb1 = np.ones([self.N_h])
            sdb2 = np.ones([N_a])

        # SAVING VARIABLES
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])

        # TRAINING LOOP BONE STRUCTURE...
        for n in tqdm(range(self.N_episodes)):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)  ## DECAYING EPSILON
            Done = 0  ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
            i = 1  ## COUNTER FOR NUMBER OF ACTIONS

            S, X, allowed_a = self.env.Initialise_game()  ## INITIALISE GAME

            # Forward pass neural network
            #Q_values, hid_layer_act = self.nn.Forwardprop_v1(X, W1, b1, W2, b2)
            Q_values, h2, x1, h1 = self.nn.Forwardprop_relu(X, W1, b1, W2, b2)

            # Get the index of the aLlowed actions
            idx_allowed, _ = np.where(allowed_a == 1)

            # Selecting the action with e-greedy
            Q_values_allowed = Q_values[idx_allowed]
            a_agent = self.h.epsilongreedy(Q_values_allowed, idx_allowed, epsilon_f)

            while Done == 0:  ## START THE EPISODE
                S_next, X_next, allowed_a_next, R, Done = self.env.OneStep(a_agent)

                ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
                if Done == 1:
                    # calculate the error
                    delta = R - Q_values[a_agent]

                    # backpropagate the error
                    #W1, W2[:, a_agent], b1, b2[a_agent] = self.nn.Backpropagation(self.eta, a_agent, delta, Q_values,
                    #                                                              hid_layer_act, X, W1, W2, b1, b2)
                    if self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_relu(self.eta, a_agent, delta, h2, x1, h1, X, W1, W2, b1, b2)
                    elif self.optimizer == "rmsprop":
                        W1, W2[:, a_agent], b1, b2[a_agent], sdw1, sdw2, sdb1, sdb2 = self.nn.Backpropagation_relu_rmsprop(self.eta, self.momentum, a_agent,
                        delta, h2, x1, h1, X, W1, W2, b1, b2, sdw1, sdw2, sdb1, sdb2)

                    self.R_save[n] = np.copy(R)
                    self.N_moves_save[n] = np.copy(i)

                    ##TWO CHOICES: if R==1, then checkmate. Else, draw.
                    break

                # IF THE EPISODE IS NOT OVER...
                else:
                    # Get the qvalues of the next state
                    #Q_values_next, _ = self.nn.Forwardprop_v1(X_next, W1, b1, W2, b2)
                    Q_values_next, _, _, _ = self.nn.Forwardprop_relu(X, W1, b1, W2, b2)

                    # selecting the qvalues of the allowed actions
                    idx_allowed_next, _ = np.where(allowed_a_next == 1)
                    Q_values_allowed_next = Q_values_next[idx_allowed_next]

                    # select the action with e-greedy
                    a_agent_next = self.h.epsilongreedy(Q_values_allowed_next, idx_allowed_next, epsilon_f)

                    # Computing the error
                    delta = R + self.gamma * np.max(Q_values_allowed_next) - Q_values[a_agent]

                    # backpropagate the error and update the weights
                    #W1, W2[:, a_agent], b1, b2[a_agent] = self.nn.Backpropagation(self.eta, a_agent, delta, Q_values,
                    #                                                              hid_layer_act, X, W1, W2, b1, b2)
                    #W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_relu(self.eta, a_agent, delta, h2, x1, h1, X, W1, W2, b1, b2)

                    if self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_relu(self.eta, a_agent, delta, h2, x1, h1, X, W1, W2, b1, b2)
                    elif self.optimizer == "rmsprop":
                        W1, W2[:, a_agent], b1, b2[a_agent], sdw1, sdw2, sdb1, sdb2 = self.nn.Backpropagation_relu_rmsprop(self.eta, self.momentum, a_agent,
                        delta, h2, x1, h1, X, W1, W2, b1, b2, sdw1, sdw2, sdb1, sdb2)

                # NEXT STATE AND CO. BECOME ACTUAL STATE...
                S = np.copy(S_next)
                X = np.copy(X_next)
                a_agent = np.copy(a_agent_next)
                allowed_a = np.copy(allowed_a_next)
                Q_values = np.copy(Q_values_next)
                i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

        print('My Agent, Average reward:', np.mean(self.R_save), 'Number of steps: ', np.mean(self.N_moves_save))

    def plot(self):
        # Storing reward and move count
        pandaR = pd.DataFrame(self.R_save)
        pandaN = pd.DataFrame(self.N_moves_save)

        # We compute the exponential moving average
        # Standard approach for setting alpha
        alpha = 2 / (self.N_episodes + 1)
        ema_r = pandaR.ewm(alpha=alpha).mean()
        ema_m = pandaN.ewm(alpha=alpha).mean()

        # Simple moving average used for low-repetition trainings
        # ema_r = pandaR.rolling(window=15).mean()
        # ema_m = pandaN.rolling(window=15).mean()

        reward_plot = ema_r.plot.line(legend=False)
        reward_plot.set_xlabel("Episode")
        reward_plot.set_ylabel("Avg. reward")

        move_plot = ema_m.plot.line(legend=False, color='red')
        move_plot.set_xlabel("Episode")
        move_plot.set_ylabel("Avg. moves")

        plt.show()