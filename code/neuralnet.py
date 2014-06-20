# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:54:28 2014

@author: ogavril
"""
import numpy as np
import math
from matplotlib.transforms import offset_copy
import matplotlib.pyplot as plt
import copy

def plot_predicted_vs_true(predicted,true):
    fig, ax = plt.subplots()
    ax.plot(predicted,true,'g.')
    #ax.set_title(stk+": Residuals vs True "+yName)
    ax.set_ylabel("True values")
    ax.set_xlabel("Predictions")
    plt.show()
    #plt.savefig(out_dir+"Resid_vs_True_"+stk+".jpg")


    
class SimpleNeuralNet:
    def __init__(self,num_features, num_hidden_neurons=4,num_epochs=500,
                 LearningRate=0.07,include_LinearNeuron = True,
                 include_InputBias=True,include_OutputBias=True): #numFeatures includes bias
        self.num_hidden = num_hidden_neurons
        self.num_epochs = num_epochs
        self.LR_IH = LearningRate
        self.LR_HO = LearningRate/10
        self.LR_IH_original = LearningRate
        self.LR_HO_original = LearningRate/10
        self.HasLinearNeuron = include_LinearNeuron
        self.num_features = num_features

        #the outputs of the hidden neurons
        self.hidden_values = [0.0 for i in range(self.num_hidden)]

        #the weights
        self.weights_IH =[[0.0 for i in range(self.num_hidden)] for j in range(self.num_features)]
        self.weights_HO = [0.0 for i in range(self.num_hidden+1)]
        self.weights_IH_best =[[0.0 for i in range(self.num_hidden)] for j in range(self.num_features)]
        self.weights_HO_best = [0.0 for i in range(self.num_hidden+1)]
        self.RMSerror_best = 1000000


    def train(self,observations,target,net_fn=None):

        self.initialize_weights()
        worsecount = 0
        for j in range(self.num_epochs):
            errors = []
            predictions = []
            targets = []
            
            #loop through all the patterns
            for i in range(observations.shape[0]):
                #select a pattern at random
                row_num = np.random.random_integers(0,observations.shape[0]-1)
    
                #calculate the current network output and error for this pattern
                error,prediction = self.calc_net(observations[row_num],target[row_num])
                errors.append(error)
                predictions.append(prediction)
                targets.append(target[row_num])
                
                #change network weights accordingly
                self.weight_changes_HO(error)
                self.weight_changes_IH(error,observations[row_num])
            
            #display the overall network error after each epoch
            RMSerror = self.calc_overall_error(errors)
            print "epoch = %d RMS Error = %f" %(j,RMSerror)
            plot_predicted_vs_true(predictions,targets)
            
            #IF ERROR HAS INCREASED THEN REVERT BACK TO STARTING WEIGHTS
            if (RMSerror < self.RMSerror_best):
                self.RMSerror_best = RMSerror
                worsecount = 0
            else:
                if worsecount > np.random.random_integers(2,5) :
                    self.weights_IH = copy.deepcopy(self.weights_IH_best)
                    self.weights_HO = copy.deepcopy(self.weights_HO_best)
#                    self.LR_HO = max(self.LR_HO*0.9,0.00007)
#                    self.LR_IH = max(self.LR_IH*0.9,0.00007)
                    worsecount += 1
#            if (worsecount > 10):
#                self.initialize_weights()        
#                self.weights_IH_best = self.weights_IH
#                self.weights_HO_best = self.weights_HO            
##                self.LR_HO = self.LR_HO_orig/2
##                self.LR_IH = self.LR_IH_orig/2
#                worsecount = 0
        
        #training has finished display the results
        #self.display_results(targets,predictions,errors)
        try:
            plot_predicted_vs_true(predictions,targets)
        except:
            print "plotting aborted due to invalid values"
        print "last RMS Error =",RMSerror
        print "number hidden neurons:",self.num_hidden
        print "learning rates are:",self.LR_IH,self.LR_HO
        return self.save_net(net_fn)

    def validate(self,observations,target):
        print "\n...validating..."
        errors = []
        predictions = []
        targets = []
        for i in range(observations.shape[0]):
            row_num = i
            #calculate the current network output and error for this pattern
            error,prediction = self.calc_net(observations[row_num],target[row_num])
            errors.append(error)
            predictions.append(prediction)
            targets.append(target[row_num])

            #change network weights
        RMSerror = self.calc_overall_error(errors)
        self.display_results(targets,predictions,errors)
        try:
            plot_predicted_vs_true(predictions,targets)
        except:
            print "plotting aborted due to invalid values"
        print "validation RMS Error = %f"%RMSerror
        return predictions
        
    def test(self,observations):
        predictions = []
        for i in range(observations.shape[0]):
            row_num = i
            #calculate the current network output and error for this pattern
            prediction = self.calc_net(observations[row_num],[],training=False)

            predictions.append(prediction)

            #change network weights
        for i in range(len(predictions)):
            print "obs =",i+1,"predicted value is",predictions[i]
        return predictions
        
    def display_results(self,targs,predictions,errors):
        for i in range(len(predictions)):
            print "actual = %.4f predicted = %.4f error = %.4f" %(targs[i],predictions[i],errors[i])
        
                    
    def calc_net(self,observations_row,target_row,training = True):
        """calculate network output: calculate the outputs of the hidden neurons; the hidden neurons are tanh """
        for i in range(self.num_hidden):
            self.hidden_values[i] = 0.0
            for j in range(self.num_features):
                self.hidden_values[i] += observations_row[j] * self.weights_IH[j][i]
            if i == 0 and self.HasLinearNeuron==True:
                self.hidden_values[i] = self.hidden_values[i] #identity
            else:
                self.hidden_values[i] = math.tanh(self.hidden_values[i])
    
       #calculate the output of the network; the output neuron is linear
        outPred = 0.0;
        for i in range(self.num_hidden):
            outPred += self.hidden_values[i]*self.weights_HO[i]
        #output bias
        outPred += self.weights_HO[self.num_hidden]
            
            
        #calculate the error
        if training:
            errThisPat = outPred - target_row
            return errThisPat,outPred
        else:
            return outPred

    def save_net(self,fn=None):
        if fn == None:
            fn = "net.net"
        fn = open(fn,'w')
        fn.write("num_hiden:"+str(self.num_hidden)+"\n")
        fn.write("num_features:"+str(self.num_features)+"\n")
        fn.write("weights_IH:\n")
        for j in range(self.num_features):
            for i in range(self.num_hidden):
                fn.write(str(self.weights_IH[j][i])+",")
            fn.write("\n")
        fn.write("\n")
        fn.write("weights_HO:\n")
        for i in range(self.num_hidden):
            fn.write(str(self.weights_HO[i])+",")
        
        return self
        
        
            
        
    def weight_changes_HO(self,error):
        """adjust the weights hidden-output"""
        for k in range(self.num_hidden):
            weight_change = self.LR_HO * error * self.hidden_values[k]
            self.weights_HO[k] += - weight_change
        self.weights_HO[self.num_hidden] += - self.LR_HO * error

        #regularisation on the output weights        
        for k in range(self.num_hidden + 1):
            if (self.weights_HO[k] < -5):
                print "ever <-5?"
                self.weights_HO[k] = -5
            elif self.weights_HO[k] > 5:
                print "ever>5?"
                self.weights_HO[k] = 5


    def weight_changes_IH(self,error,observations_row):
        """adjust the weights input-hidden"""
        for i in range(self.num_hidden):
            for k in range(len(observations_row)):
                #Phil said not to do the commented out lines because weights can get too big
#                if i == 0 and self.HasLinearNeuron==True:
#                    weight_change = 1
#                else: #nonlinear change
#                    weight_change = 1 - (self.hidden_values[i] **2)                   
                weight_change = 1 - (self.hidden_values[i] **2)    
                weight_change = weight_change * self.weights_HO[i] * error * self.LR_IH
                weight_change = weight_change * observations_row[k]
                self.weights_IH[k][i] = self.weights_IH[k][i] - weight_change

                if (self.weights_IH[k][i] < -30):
                    self.weights_IH[k][i] = -30
                    print "large weights!"
                elif (self.weights_IH[k][i] > 30):
                    self.weights_IH[k][i] = 30
                    print "large weights!"



    def initialize_weights(self):
        """ set weights to random numbers """
        for j in range(self.num_hidden):
            self.weights_HO[j] = (np.random.uniform(0,1) - 0.5)/2.0
            for i in range(self.num_features):
                self.weights_IH[i][j] = (np.random.uniform(0,1)- 0.5)/5.0
                #print "Weight =",self.weights_IH[i][j]
                #set up output bias initial weight
        self.weights_HO[self.num_hidden] = (np.random.uniform(0,1) - 0.5)/2.0
        #copy them
        self.weights_IH_best = copy.deepcopy(self.weights_IH)
        self.weights_HO_best = copy.deepcopy(self.weights_HO)



    def calc_overall_error(self,errors):
        """calculate the overall error"""
        RMSerror = sum([error**2 for error in errors])

        RMSerror = math.sqrt(RMSerror/len(errors))
        return RMSerror
        
        
        
##### IGNORE below this line ... for testing purposes only  ##################
"""did these for testing purposes only"""
def calc_net(net,observations_row,target_row):
    """calculate network output: calculate the outputs of the hidden neurons; the hidden neurons are tanh """
    for i in range(net.num_hidden):
        net.hidden_values[i] = 0.0
        for j in range(net.num_features):
            net.hidden_values[i] += observations_row[j] * net.weights_IH[j][i]
        net.hidden_values[i] = math.tanh(net.hidden_values[i])

   #calculate the output of the network; the output neuron is linear
    outPred = 0.0;
    for i in range(net.num_hidden):
        outPred += net.hidden_values[i]*net.weights_HO[i]
    #calculate the error
    errThisPat = outPred - target_row
    return errThisPat,outPred

def validate(observations,target,net):
    errors = []
    predictions = []
    for i in range(observations.shape[0]):
        #select a pattern at random
        row_num = i#np.random.random_integers(0,observations.shape[0]-1)

        #calculate the current network output and error for this pattern
        error,prediction = calc_net(net, observations[row_num],target[row_num])
        errors.append(error)
        predictions.append(prediction)

        #change network weights

    RMSerror = math.sqrt(sum(error**2 for error in errors)/len(errors))

    print "RMS Error = %f\n" %RMSerror
    #training has finished display the results

    for i in range(len(predictions)):
        print "obs =",i+1,"actual value is",target[i], "predicted value is",predictions[i]

    print RMSerror    
