# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:13:51 2014

@author: olenaG

PURPOSE: this is my attempt to start using NN for data where there are categorical variables and boolean/categorical response


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuralnet
import functions

def hist_plot(vec,title=None,figname=None):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist(vec,bins=50)
    if title != None:
        fig.suptitle(title)
    plt.show()
#    if figname == None:
#        plt.savefig('hist.jpg')
#    else:
#        plt.savefig(figname+".jpg")

def write_stats_2file(train_fn,Models,Train_Stats,Validation_Stats,train_data,validation_data,ofn=None):
    if ofn == None:
        ofn = train_fn+"_stats.csv"
    ofn  = open(ofn,"w")
    kernels = Models.keys()
    headers =  Models[kernels[0]].keys()
    headers.remove("model")
    ofn.write("dataName,numObs,"+",".join(headers)+",tr_RMSE,val_RMSE,tr_sensitivity,tr_precision,tr_specificiy,tr_accuracy,val_sensitivity,val_precision,val_specificiy,val_accuracy\n")
    for kernel in kernels:
        s =train_fn+","+str(train_data.shape[0])+","
        for header in headers:
            s += str(Models[kernel][header])+","
        s+= str(round(Train_Stats[kernel]['RMSE'],4)) +","
        s+= str(round(Validation_Stats[kernel]['RMSE'],4)) +","
        for stat in ['sensitivity','precision','specificity','accuracy']:
            s+= str(round(Train_Stats[kernel][stat],2))+","
        for stat in ['sensitivity','precision','specificity','accuracy']:
            s+= str(round(Validation_Stats[kernel][stat],2))+","
            
        ofn.write(s+"\n")
    ofn.close()

def transform_into_binary(nparray,TH):
    t = []
    for elem in nparray:
        if elem < TH:
            t.append(-1)
        else:
            t.append(1)
    return np.array(t)
    
def calculate_confusion_matrix(predicted_values_train,train_target,predicted_values_validation,validation_target,logf):
    TH = 0.0
    transformed_predicted_values_train = transform_into_binary(predicted_values_train,TH)
    transformed_predicted_values_validation = transform_into_binary(predicted_values_validation,TH)
    tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = functions.calculate_SensSpecifPrecAccurNN(transformed_predicted_values_train,train_target)
    Train_Stats[kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
    ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = functions.calculate_SensSpecifPrecAccurNN(transformed_predicted_values_validation,validation_target)                
    Validation_Stats[kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}  
    s =  kernel+"\n"
    s += "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
    s += "For the validation set "+str(validation_set)+" of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
    logf.write(s+"\n\n")
    
    
def initialize_train_data(DATA_DIR,fn):
    df = pd.read_csv(DATA_DIR+fn)
    """needs possible extensions: 1) filling NANs, 2) trimming; 3) outliering, 4) transforming, etc."""
    return df
    
def initialize_test_data(DATA_DIR,fn):
    """usually differs from train data as there is target"""
    df = pd.read_csv(DATA_DIR+fn)
    """needs possible extensions: 1) filling NANs, 2) trimming; 3) outliering, 4) transforming, etc."""
    return df


def valid_variables(df,target_name,categorical_vars):
    t = []
    for col in df.columns:
        if col not in [ "_TRAIN"]+categorical_vars+[target_name]:
            if not col.startswith("_"):
                t.append(col)
    return t
    
def valid_nonCategorical_variables(df,valid_columns):
    t = []
    for col in valid_columns:
        if col.find(":") < 0:
                t.append(col)
    return t
    

if __name__== "__main__":
    logf = open("log_voteNN.log",'w')
    #load in the data
    DATA_DIR = ".."+os.sep+"data"+os.sep
    train_fn = "test08_BinaryResponse.csv"

    train_df = initialize_train_data(DATA_DIR,train_fn)
    test_fn = None #"test01b.csv" #None

    target_var = "vote"

    """organize data into train, test,and vaildate"""
    train_var = "_TRAIN"
    validation_set = 4
    test_set = -1#validation_set +1

    #if there is no test file, then make a test data set out of train, but using a bit of it
    train_df = functions.def_cross_validation_subsets(train_df,train_var,numK=validation_set+1)
    test_df = train_df[train_df[train_var] == test_set]
    train_df = train_df[train_df[train_var] != test_set]

    """put the two DFs together to perform transformations, trimming, filling NANs if necessary etc."""        
    DF = pd.concat([train_df, test_df], ignore_index=False)
    DF['const'] = 1.0 #adding the bias node; in some situations it should be omitted
    print "size of concatenated DF",len(DF),"number of columns:", len(DF.columns)
    
    categorical_vars = []
    explanatory_vars = valid_variables(train_df,target_var,categorical_vars)
    if 'const' in DF.columns:
        explanatory_vars += ['const']
    print "useful vars:",explanatory_vars

    scaled_DF = DF.copy()
    for col in explanatory_vars:
        scaled_DF[col] = functions.scale(DF,col)
    #scaled_DF.to_csv("scaledDF.csv")
    
    scaled_DF[target_var] = functions.scale(DF,target_var)
    
    """separate the two DFs AFTER all the variable manipulating work is done"""
    train_df = scaled_DF[scaled_DF[train_var] != test_set ] 
    test_df = scaled_DF[scaled_DF[train_var] == test_set]

    train_data = functions.make_numpy_matrix(train_df[train_df[train_var] != validation_set],explanatory_vars)
    train_target = np.array(train_df[target_var][train_df[train_var] != validation_set])#.reshape(train_data.shape[0],1)

    validation_data = functions.make_numpy_matrix(train_df[train_df[train_var] == validation_set],explanatory_vars)
    validation_target = np.array(train_df[target_var][train_df[train_var] == validation_set])#.reshape(validation_data.shape[0],1)
    
#    test_data = make_numpy_matrix(test_df,explanatory_vars)
#    test_target = np.array(test_df[target_var])#.reshape(test_data.shape[0],1)
                
    
    #neural_net.test(test_data)
    Train_Stats = {}
    Validation_Stats = {}
    Models = {}
    kernel = "NN"
    MaxNumHiddenNeurons = train_data.shape[1]+3 # int(1.5*train_data.shape[1])+1
    MaxNumEpochs = 550
    LearningRates = [0.005] #0.05,0.005]

    for hd in range(train_data.shape[1]+1,MaxNumHiddenNeurons,1):
        for numEpochs in range(500,MaxNumEpochs,100):
            for lr in LearningRates:
                for linNeuron in [True,False]:                   
                            
                    neural_net = neuralnet.SimpleNeuralNet(train_data.shape[1],num_hidden_neurons=hd, 
                                                           num_epochs=numEpochs,LearningRate=lr,include_LinearNeuron = linNeuron,
                                                           include_InputBias=True,include_OutputBias=True)
                    net = neural_net.train(train_data,train_target,plot=False)
                    #print "weights_HO:",net.weights_HO
                    #print "weights_HI:",net.weights_IH
                    
                    predicted_values_train,RMSE_train = neural_net.validate(train_data,train_target) 
                    predicted_values_validation,RMSE_validation= neural_net.validate(validation_data,validation_target)
                    
                    kernel = "NN_"+"NumHiddenNeurons:"+str(hd)+"_NumEpochs:"+str(numEpochs)+"_LR:"+str(lr)+"_LinNeuron:"+str(linNeuron)
                    if kernel not in Models.keys():
                        Train_Stats[kernel] = {}
                        Validation_Stats[kernel] = {}
                        Models[kernel] = {}
                    Models[kernel]['model'] = net
                    Models[kernel]["NumHiddenNeurons"] = hd
                    Models[kernel]["NumEpochs"]= numEpochs
                    Models[kernel]["StartingLearningRate"]=lr
                    Models[kernel]["IncludeLinearNeuron"]=linNeuron
                    Models[kernel]["ActivationFunctions"] = "tanh"
                    Models[kernel]["NumFeatures"] = train_data.shape[1]
                    Train_Stats[kernel]['RMSE'] = RMSE_train
                    Validation_Stats[kernel]["RMSE"] = RMSE_validation 
                    hist_plot(predicted_values_train,title="TrainSet Prediction")
                    hist_plot(predicted_values_validation,title="ValidationSet Prediction")
                    calculate_confusion_matrix(predicted_values_train,train_target,predicted_values_validation,validation_target,logf)



    write_stats_2file(train_fn,Models,Train_Stats,Validation_Stats,train_data,validation_data)  
    logf.close()
   
