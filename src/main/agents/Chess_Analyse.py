import numpy as np
from env.Chess_env import Chess_Env

class Analyse_Env:
    def __init__(self):
        # initialize Chess 4x4 env
        self.env = Chess_Env(4)
    
    def analyse(self):
        # analyse chess board for better understanding

        S,X,allowed_a=self.env.Initialise_game()
        print(S)

        # PRINT VARIABLE THAT TELLS IF ENEMY KING IS IN CHECK (1) OR NOT (0)
        print('check?',self.env.check) 

        # PRINT THE NUMBER OF LOCATIONS THAT THE ENEMY KING CAN MOVE TO
        print('dofk2 ',np.sum(self.env.dfk2_constrain).astype(int))

        for i in range(5):
            a,_=np.where(allowed_a==1)                  # FIND WHAT THE ALLOWED ACTIONS ARE
            a_agent=np.random.permutation(a)[0]         # MAKE A RANDOM ACTION

            S,X,allowed_a,R,Done=self.env.OneStep(a_agent)   # UPDATE THE ENVIRONMENT

            ## PRINT CHESS BOARD AND VARIABLES
            print('')
            print(S)
            print(R,'', Done)
            print('check? ',self.env.check)
            print('dofk2 ',np.sum(self.env.dfk2_constrain).astype(int))
            
            # TERMINATE THE EPISODE IF Done=True (DRAW OR CHECKMATE)
            if Done:
                break