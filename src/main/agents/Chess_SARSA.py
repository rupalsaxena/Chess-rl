import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from env.Chess_env import Chess_Env
from env.generate_game import generate_game
from utils.NN import NN
from utils.helpers import helpers

class Chess_SARSA:
    def __init__(self, config) -> None:
        self.N_h = config["N_h"]
        self.N_episodes = config["N_episodes"]
        self.epsilon_0 = config["epsilon_0"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.s = config["s"]
        self.Reward_Check = config["Reward_Check"]
        self.xavier = config["xavier_init"]
        self.activation = config["activation"]
        self.optimizer = config["optimizer"]
        self.nn = NN()
        self.h = helpers()

        # initialize 4x4 chess board env
        self.env = Chess_Env(4)

    def train(self):
        S,X,allowed_a=self.env.Initialise_game()
        N_a=np.shape(allowed_a)[0]# TOTAL NUMBER OF POSSIBLE ACTIONS

        N_in=np.shape(X)[0]    ## INPUT SIZE

        #Initialise random seeded array
        np.random.seed(self.s)

        ## INITALISE YOUR NEURAL NETWORK...

        ##Xavier init
        if self.xavier:
            W1=np.random.randn(N_in, self.N_h)*np.sqrt(1/(N_in))
            W2=np.random.randn(self.N_h, N_a)*np.sqrt(1/self.N_h)
        else:
            W1=np.random.uniform(0,300,[N_in,self.N_h])/(N_in+self.N_h)
            W2=np.random.uniform(0,300,[self.N_h,N_a])/(N_in+self.N_h)

        b1=np.zeros([self.N_h])
        b2=np.zeros([N_a])

        # SAVING VARIABLES
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])

        # TRAINING LOOP BONE STRUCTURE...
        for n in tqdm(range(self.N_episodes)):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)   ## DECAYING EPSILON
            Done=0                                             ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
            i = 1                                              ## COUNTER FOR NUMBER OF ACTIONS

            S,X,allowed_a=self.env.Initialise_game()           ## INITIALISE GAME

            #Forward pass neural network
            if self.activation == "relu":
                Q_values, out_layer, hid_layer_act, hid_layer = self.nn.Forwardprop_relu(X, W1, b1, W2, b2)
            elif self.activation == "sigmoid":
                Q_values, hid_layer_act = self.nn.Forwardprop_sigmoid(X, W1, b1, W2, b2)
            else:
                raise Exception("This activation function not implemented in Neural Network!")
    
            #Get the index of the aLlowed actions
            idx_allowed,_=np.where(allowed_a==1)

            #Selecting the action with e-greedy
            Q_values_allowed=Q_values[idx_allowed]
            a_agent = self.h.epsilongreedy(Q_values_allowed, idx_allowed, epsilon_f)

            while Done==0:                                   ## START THE EPISODE
                S_next,X_next,allowed_a_next,R,Done=self.env.OneStep(a_agent)

                ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
                if Done==1:
                    ##For the change of the representation of the reward
                    ##Uncomment the if and else to augment reward conditionnaly on a short path
                    if R==1:
                        #if i<=5:
                        Rw = self.Reward_Check
                        #else: Rw = 1
                    else: 
                        Rw = 0

                    #calculate the error
                    delta=Rw-Q_values[a_agent]

                    #backpropagate the error
                    if self.activation == "relu" and self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_relu(self.eta, a_agent, delta, out_layer, hid_layer_act, hid_layer, X, W1, W2, b1, b2)
                    elif self.activation == "sigmoid" and self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_sigmoid(self.eta, a_agent, delta, Q_values, hid_layer_act, X, W1, W2, b1, b2)
                    else:
                        raise Exception("This activation function not implemented in Neural Network!")

                    self.R_save[n]=np.copy(R)
                    self.N_moves_save[n]=np.copy(i)

                    ##TWO CHOICES: if R==1, then checkmate. Else, draw.
                    break

                # IF THE EPISODE IS NOT OVER...
                else:
                    #Get the qvalues of the next state
                    if self.activation == "relu":
                        Q_values_next, _, _, _ = self.nn.Forwardprop_relu(X_next, W1, b1, W2, b2)
                    elif self.activation == "sigmoid":
                        Q_values_next, _ = self.nn.Forwardprop_sigmoid(X_next, W1, b1, W2, b2)
                    else:
                        raise Exception("This activation function not implemented in Neural Network!")
    
                    #selecting the qvalues of the allowed actions
                    idx_allowed_next,_=np.where(allowed_a_next==1)
                    Q_values_allowed_next=Q_values_next[idx_allowed_next]

                    #select the action with e-greedy
                    a_agent_next = self.h.epsilongreedy(Q_values_allowed_next, idx_allowed_next, epsilon_f)

                    #Computing the error
                    delta=R+self.gamma*Q_values_next[a_agent_next]-Q_values[a_agent]

                    #backpropagate the error and update the weights
                    if self.activation == "relu" and self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent] = self.nn.Backpropagation_relu(self.eta, a_agent, delta, out_layer, hid_layer_act, hid_layer, X, W1, W2, b1, b2)
                    elif self.activation == "sigmoid" and self.optimizer == "gd":
                        W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation_sigmoid(self.eta, a_agent, delta, Q_values, hid_layer_act, X, W1, W2, b1, b2)
                    else:
                        raise Exception("This activation function not implemented in Neural Network!")

                # NEXT STATE AND CO. BECOME ACTUAL STATE...
                S=np.copy(S_next)
                X=np.copy(X_next)
                a_agent = np.copy(a_agent_next)
                allowed_a = np.copy(allowed_a_next)
                Q_values = np.copy(Q_values_next)
                i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

        print('My Agent, Average reward:',np.mean(self.R_save),'Number of steps: ',np.mean(self.N_moves_save))

    def plot(self):
        pandaR = pd.DataFrame(self.R_save)
        pandaN = pd.DataFrame(self.N_moves_save)
        ema_r = pandaR.ewm(alpha=0.001, adjust = False).mean()
        ema_m = pandaN.ewm(alpha=0.001, adjust = False).mean()
        time=np.arange(1, (len(self.R_save)+1))

        ##Only one at a time and no subplot would look better but in the meantime
        plt.subplot(2, 1, 1)
        plt.scatter(time, ema_m)
        plt.xlabel("Time")
        plt.ylabel("Average number of moves")
        plt.subplot(2, 1, 2)
        plt.scatter(time, ema_r)
        plt.xlabel("Time")
        plt.ylabel("Average reward")
        plt.show()
