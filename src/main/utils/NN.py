import numpy as np
from utils.helpers import helpers

class NN:
    def __init__(self):
        self.h = helpers()

    def Forwardprop(self, x, W1, b1, W2, b2):

        hid_layer = np.dot(x, W1)+b1
        hid_layer_act = np.maximum(hid_layer, 0)
        out_layer=np.dot(hid_layer_act, W2)+b2
        out_layer_act =np.maximum(out_layer, 0)
        
        return out_layer_act, hid_layer_act
    
    ##done similarly to the "chess student" file
    #W2 : only modifying the weights going to a_agent
    #W1 : as the features are represented with all the 58 entries (unlike the action taken), all weights are updated (?)
    def Backpropagation(self, eta, a_agent, delta, out_layer_act, hid_layer_act, x,  W1, W2, b1, b2):
        
        delta_W2=eta*delta*out_layer_act[a_agent]
        delta_W1=eta*np.outer(x, (delta*W2[:, a_agent]*(self.h.reluder(hid_layer_act))))

        W2[:,a_agent]=W2[:,a_agent]+eta*delta_W2
        b2[a_agent]=b2[a_agent]+eta*delta
        
        W1=W1+eta*delta_W1
        b1=b1+eta*delta*(W2[:,a_agent])*self.h.reluder(hid_layer_act)
        

        return  W1,  W2[:, a_agent], b1, b2[a_agent]
