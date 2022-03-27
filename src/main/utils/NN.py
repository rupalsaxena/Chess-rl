import numpy as np
from utils.helpers import helpers

class NN:
    def __init__(self):
        self.h = helpers()

    def Forwardprop_v1(self, x, W1, b1, W2, b2):
        hid_layer = np.dot(x, W1)+b1
        hid_layer_act = np.maximum(hid_layer, 0)
        out_layer=np.dot(hid_layer_act, W2)+b2
        out_layer_act =np.maximum(out_layer, 0)
        return out_layer_act, hid_layer_act
    
    def Forwardprop_relu(self, x, W1, b1, W2, b2):
        h1 = np.dot(x, W1)+b1 
        x1= np.maximum(h1, 0)
        h2=np.dot(x1, W2)+b2
        x2 =np.maximum(h2, 0)
        return x2, h2, x1, h1
    
    def Forwardprop_sigmoid(self, x, W1, b1, W2, b2):
        h1 = np.dot(x, W1)+b1 
        x1= 1/(1+np.exp(-h1))
        h2=np.dot(x1, W2)+b2
        x2 = 1/(1+np.exp(-h2))
        return x2, x1

    ###TO REMOVE
    ##done similarly to the "chess student" file 
    #W2 : only modifying the weights going to a_agent
    #W1 : as the features are represented with all the 58 entries (unlike the action taken), all weights are updated (?)
    def BackpropagationFirstVersion(self, eta, a_agent, delta, out_layer_act, hid_layer_act, x,  W1, W2, b1, b2):
        delta_W2=delta*out_layer_act[a_agent]
        delta_W1=np.outer(x, (delta*W2[:, a_agent]*(self.h.reluder(hid_layer_act))))

        W2[:,a_agent]=W2[:,a_agent]+eta*delta_W2
        b2[a_agent]=b2[a_agent]+eta*delta

        W1=W1+eta*delta_W1
        b1=b1+eta*delta*(W2[:,a_agent])*self.h.reluder(hid_layer_act)


        return  W1,  W2[:, a_agent], b1, b2[a_agent]

    ##Adapted from the Lab 1 of the Reinforcement Learning lecture
    ##Corrected!
    def Backpropagation(self, eta, a_agent, delta, out_layer_act, hid_layer_act, x,  W1, W2, b1, b2):
        delta_W2=delta*hid_layer_act*(out_layer_act[a_agent]>0)
        delta_W1=np.outer(x, (delta*(out_layer_act[a_agent]>0)*W2[:,a_agent]*(hid_layer_act>0)))

        W2[:, a_agent]=W2[:, a_agent]+eta*delta_W2
        b2[a_agent]=b2[a_agent]+eta*delta*(out_layer_act[a_agent]>0)
       
        W1=W1+eta*delta_W1
        b1=b1+eta*delta*(out_layer_act[a_agent]>0)*W2[:,a_agent]*(hid_layer_act>0)
        return  W1,  W2[:, a_agent], b1, b2[a_agent]

    def Backpropagation_relu(self, eta, a_agent, delta, out_layer, hid_layer_act, hid_layer, x,  W1, W2, b1, b2):
        # gradient descent optimizer
        delta_W2 = delta*hid_layer_act*(out_layer[a_agent]>0)
        delta_W1 = np.outer(x, (delta*(out_layer[a_agent]>0)*W2[:,a_agent]*(hid_layer>0)))
        W2[:, a_agent]=W2[:, a_agent]+eta*delta_W2
        b2[a_agent]=b2[a_agent]+eta*delta*(out_layer[a_agent]>0)

        W1=W1+eta*delta_W1
        b1=b1+eta*delta*(out_layer[a_agent]>0)*W2[:,a_agent]*(hid_layer>0)
        return  W1,  W2[:, a_agent], b1, b2[a_agent]
    
    def Backpropagation_relu_rmsprop(self, eta, beta, a_agent, delta, out_layer, hid_layer_act, hid_layer, x,  W1, W2, b1, b2, sdw1, sdw2, sdb1, sdb2):
        #rms prop optimizer
        delta_W2 = delta*hid_layer_act*(out_layer[a_agent]>0)
        delta_W1 = np.outer(x, (delta*(out_layer[a_agent]>0)*W2[:,a_agent]*(hid_layer>0)))

        delta_b2 = delta*(out_layer[a_agent]>0)
        delta_b1 = delta*(out_layer[a_agent]>0)*W2[:,a_agent]*(hid_layer>0)

        sdw2 = beta*sdw2 + (1-beta)*(delta_W2**2)
        sdw1 = beta*sdw1 + (1-beta)*(delta_W1**2)

        sdb2 = beta*sdb2 + (1-beta)*(delta_b2**2)
        sdb1 = beta*sdb1 + (1-beta)*(delta_b1**2)

        W2[:, a_agent]=W2[:, a_agent]+(eta*delta_W2)/np.sqrt(sdw2)
        b2[a_agent]=b2[a_agent]+(eta*delta_b2)/np.sqrt(sdb2[a_agent])

        W1=W1+(eta*delta_W1)/np.sqrt(sdw1)
        b1=b1+(eta*delta_b1)/np.sqrt(sdb1)
        return  W1,  W2[:, a_agent], b1, b2[a_agent], sdw1, sdw2, sdb1, sdb2
    
    def Backpropagation_sigmoid(self, eta, a_agent, delta, x2, x1, x,  W1, W2, b1, b2):
        # gradient descent optimizer
        delta2 = x2[a_agent]*(1-x2[a_agent])
        delta_W2 = x1*delta2
        
        W2[:, a_agent]=W2[:, a_agent]-eta*delta*delta_W2
        b2[a_agent]=b2[a_agent]-eta*delta*delta2
        
        delta1 = delta * x1*(1-x1) * W2[:, a_agent] * delta2
        delta_W1 = np.outer(x,delta1)

        W1=W1-eta*delta_W1
        b1=b1-eta*delta1
        return  W1,  W2[:, a_agent], b1, b2[a_agent]