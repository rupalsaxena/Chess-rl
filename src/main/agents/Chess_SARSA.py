import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        self.eligibility_trace = config["eligibility_trace"]

        self.nn = NN()
        self.h = helpers()

        # initialize 4x4 chess board env
        self.env = Chess_Env(4)

    def train(self):
        S,X,allowed_a=self.env.Initialise_game()
        N_a=np.shape(allowed_a)[0]# TOTAL NUMBER OF POSSIBLE ACTIONS

        N_in=np.shape(X)[0]    ## INPUT SIZE

        ## INITALISE YOUR NEURAL NETWORK...
        W1=np.random.uniform(0,0,[N_in,self.N_h])/(N_in+self.N_h)
        b1=np.zeros([self.N_h])

        W2=np.random.uniform(0,0,[self.N_h,N_a])/(N_in+self.N_h)
        b2=np.zeros([N_a])

        # SAVING VARIABLES
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])

        #Added Eligibility Trace; parameters taken from lab 3
        if self.eligibility_trace:
            lamb=0.3
            self.eta=0.08

        # TRAINING LOOP BONE STRUCTURE...
        for n in range(self.N_episodes):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)   ## DECAYING EPSILON
            Done=0                                             ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
            i = 1                                              ## COUNTER FOR NUMBER OF ACTIONS

            S,X,allowed_a=self.env.Initialise_game()           ## INITIALISE GAME

            #Forward pass neural network
            Q_values, hid_layer_act = self.nn.Forwardprop(X, W1, b1, W2, b2)

            #Get the index of the aLlowed actions
            idx_allowed,_=np.where(allowed_a==1)

            #Selecting the action with e-greedy
            Q_values_allowed=Q_values[idx_allowed]
            a_agent = self.h.epsilongreedy(Q_values_allowed, idx_allowed, epsilon_f)

            # Initialise eligibility traces
            #e1: everything gets updated
            #e2: only the action taken gets updated
            if self.eligibility_trace==True:
                e1=np.zeros([N_in,self.N_h])
                e2=np.zeros([self.N_h, N_a])

            while Done==0:                                   ## START THE EPISODE
                if self.eligibility_trace:
                    e1=e1+1
                    e2[:, a_agent]=e2[:, a_agent]+1

                S_next,X_next,allowed_a_next,R,Done=self.env.OneStep(a_agent)

                ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
                if Done==1:
                    #calculate the error
                    delta=R-Q_values[a_agent]

                    #backpropagate the error
                    W1, W2[:, a_agent], b1, b2[a_agent]= self.nn.Backpropagation(self.eta, a_agent, delta, Q_values, hid_layer_act, X, W1, W2, b1, b2)

                    if self.eligibility_trace:
                        W1=W1+self.eta*delta*e1
                        W2[:, a_agent]=W2[:, a_agent]+self.eta*delta*e2[:, a_agent]

                    self.R_save[n]=np.copy(R)
                    self.N_moves_save[n]=np.copy(i)

                    ##TWO CHOICES: if R==1, then checkmate. Else, draw.
                    break

                # IF THE EPISODE IS NOT OVER...
                else:
                    #Get the qvalues of the next state
                    Q_values_next, _ = self.nn.Forwardprop(X_next, W1, b1, W2, b2)

                    #selecting the qvalues of the allowed actions
                    idx_allowed_next,_=np.where(allowed_a_next==1)
                    Q_values_allowed_next=Q_values_next[idx_allowed_next]

                    #select the action with e-greedy
                    a_agent_next = self.h.epsilongreedy(Q_values_allowed_next, idx_allowed_next, epsilon_f)

                    #Computing the error
                    delta=R+self.gamma*Q_values_next[a_agent_next]-Q_values[a_agent]

                    #backpropagate the error and update the weights
                    W1, W2[:, a_agent], b1, b2[a_agent] = self.nn.Backpropagation(self.eta, a_agent, delta, Q_values, hid_layer_act, X, W1, W2, b1, b2)

                    if self.eligibility_trace:
                        W1=W1+self.eta*delta*e1
                        e1=self.gamma*lamb*e1
                        W2[:, a_agent]=W2[:, a_agent]+self.eta*delta*e2[:, a_agent]
                        e2=self.gamma*lamb*e2

                # NEXT STATE AND CO. BECOME ACTUAL STATE...
                S=np.copy(S_next)
                X=np.copy(X_next)
                a_agent = np.copy(a_agent_next)
                allowed_a = np.copy(allowed_a_next)
                Q_values = np.copy(Q_values_next)
                i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS
        print('My Agent, Average reward:',np.mean(self.R_save),'Number of steps: ',np.mean(self.N_moves_save))

    def plot(self):
        import pandas as pd
        pandaR = pd.DataFrame(self.R_save)
        pandaN = pd.DataFrame(self.N_moves_save)
        ema_r = pandaR.ewm(alpha=0.0001, adjust = False).mean()
        ema_m = pandaN.ewm(alpha=0.0001, adjust = False).mean()
        time=np.arange(1, (len(R_save)+1))
        ##Only one at a time and no subplot would look better but in the meantime  
        plt.subplot(2, 1, 1)
        plt.scatter(time, ema_m)
        plt.subplot(2, 1, 2)
        plt.scatter(time, ema_r)
    
   
