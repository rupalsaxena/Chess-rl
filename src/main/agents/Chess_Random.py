import numpy as np
from env.Chess_env import Chess_Env

class Chess_Random:
    def __init__(self):
        # initialize 4x4 chess env
        self.env = Chess_Env(4)

    def train(self):
        # PERFORM N_episodes=1000 EPISODES MAKING RANDOM ACTIONS AND COMPUTE THE AVERAGE REWARD AND NUMBER OF MOVES 
        S,X,allowed_a=self.env.Initialise_game()
        N_episodes=1000

        # VARIABLES WHERE TO SAVE THE FINAL REWARD IN AN EPISODE AND THE NUMBER OF MOVES 
        R_save_random = np.zeros([N_episodes, 1])
        N_moves_save_random = np.zeros([N_episodes, 1])

        for n in range(N_episodes):
            S,X,allowed_a=self.env.Initialise_game()     # INITIALISE GAME
            Done=0                                  # SET Done=0 AT THE BEGINNING
            i=1                                     # COUNTER FOR THE NUMBER OF ACTIONS (MOVES) IN AN EPISODE
            # UNTIL THE EPISODE IS NOT OVER...(Done=0)
            while Done==0:
                # SAME AS THE CELL BEFORE, BUT SAVING THE RESULTS WHEN THE EPISODE TERMINATES 
                a,_=np.where(allowed_a==1)
                a_agent=np.random.permutation(a)[0]
                S,X,allowed_a,R,Done=self.env.OneStep(a_agent)
                
                if Done:
                    R_save_random[n]=np.copy(R)
                    N_moves_save_random[n]=np.copy(i)
                    break
                i=i+1                               # UPDATE THE COUNTER
        # AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT, SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW 
        # (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)
        print('Random_Agent, Average reward:',np.mean(R_save_random),'Number of steps: ',np.mean(N_moves_save_random))
