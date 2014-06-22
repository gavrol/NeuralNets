# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:13:51 2014

@author: ogavril
"""

import os
import numpy as np
import pandas as pd

import neuralnet
import functions

def write_stats_2file(train_fn,Models,Train_Stats,Validation_Stats,train_data,validation_data,ofn=None):
    if ofn == None:
        ofn = train_fn+"_stats.csv"
    ofn  = open(ofn,"w")
    kernels = Models.keys()
    headers =  Models[kernels[0]].keys()
    headers.remove("model")
    ofn.write("dataName,numObs,"+",".join(headers)+",trainRMSE,validateRMSE"+"\n")
    for kernel in kernels:
        s =train_fn+","+str(train_data.shape[0])+","
        for header in headers:
            s += str(Models[kernel][header])+","
        s+= str(round(Train_Stats[kernel]['RMSE'],4)) +","
        s+= str(round(Validation_Stats[kernel]['RMSE'],4)) +","
        ofn.write(s+"\n")
    ofn.close()

def scale(df,col):
    tmp = []
    min_ = df[col].min()
    max_ = df[col].max()
    if min_ != max_:
        for indx in df.index:
            tmp.append((df[col][indx] - min_)/(max_ - min_)*2 - 1)
        return tmp
    else:
        return np.ones(len(df.index))

    
def make_numpy_matrix(df,variables):
    """assumes that the bias was already added"""
    observations = []
    for col in variables:
        observations.append(np.array(df[col]))
    observations = np.mat(observations).transpose().A #annoying numpy magic, and Tim loves it
    print observations.shape
    return observations    
    
def initialize_train_data(DATA_DIR,fn):
    df = pd.read_csv(DATA_DIR+fn)
    """needs possible extensions: 1) filling NANs, 2) trimming; 3) outliering, 4) transforming, etc."""
    return df
    
def initialize_test_data(DATA_DIR,fn):
    """usually differs from train data as there is target"""
    df = pd.read_csv(DATA_DIR+fn)
    """needs possible extensions: 1) filling NANs, 2) trimming; 3) outliering, 4) transforming, etc."""
    return df

def valid_variables(df,target_var):
    col_names = []
    for col in df.columns:
        if col not in ["_TRAIN",target_var]:
            col_names.append(col)
    return col_names

if __name__== "__main__":

    #load in the data
    DATA_DIR = ".."+os.sep+"data"+os.sep
    train_fn = "test07_SquarePlusItself.csv"#"test04.csv"#"test03_random.csv"#"test05.csv"#"test02_logNexp.csv"#"test06_Square_plus_random.csv"#"test07_Square.csv"#  "test01.csv"# 
    
    #observations,target,df = initialize_data(DATA_DIR,train_fn)
    train_df = initialize_train_data(DATA_DIR,train_fn)
    test_fn = None #"test01b.csv" #None

    target_var = "response"

    """organize data into train, test,and vaildate"""
    train_var = "_TRAIN"
    validation_set = 6
    test_set = -1#validation_set +1
    if test_fn == None:
        #if there is no test file, then make a test data set out of train, but using a bit of it
        train_df = functions.def_cross_validation_subsets(train_df,train_var,numK=validation_set+2)
        test_df = train_df[train_df[train_var] == test_set]
        train_df = train_df[train_df[train_var] != test_set]
    else:
        test_df = initialize_test_data(DATA_DIR,test_fn)
        train_df = functions.def_cross_validation_subsets(train_df,train_var,numK=validation_set+1)
        test_df[train_var] = test_set

    """put the two DFs together to perform transformations, trimming, filling NANs if necessary etc."""        
    DF = pd.concat([train_df, test_df], ignore_index=False)
    DF['const'] = 1.0 #adding the bias node; in some situations it should be omitted
    print "size of concatenated DF",len(DF),"number of columns:", len(DF.columns)
    
    explanatory_vars = valid_variables(train_df,target_var)
    if 'const' in DF.columns:
        explanatory_vars += ['const']
    print "useful vars:",explanatory_vars

    scaled_DF = DF.copy()
    for col in explanatory_vars:
        scaled_DF[col] = scale(DF,col)
    #scaled_DF.to_csv("scaledDF.csv")
    
    scaled_DF[target_var] = scale(DF,target_var)
    
    """separate the two DFs AFTER all the variable manipulating work is done"""
    train_df = scaled_DF[scaled_DF[train_var] != test_set ] 
    test_df = scaled_DF[scaled_DF[train_var] == test_set]

    train_data = make_numpy_matrix(train_df[train_df[train_var] != validation_set],explanatory_vars)
    train_target = np.array(train_df[target_var][train_df[train_var] != validation_set])#.reshape(train_data.shape[0],1)

    validation_data = make_numpy_matrix(train_df[train_df[train_var] == validation_set],explanatory_vars)
    validation_target = np.array(train_df[target_var][train_df[train_var] == validation_set])#.reshape(validation_data.shape[0],1)
    
#    test_data = make_numpy_matrix(test_df,explanatory_vars)
#    test_target = np.array(test_df[target_var])#.reshape(test_data.shape[0],1)
                
    
    #neural_net.test(test_data)
    Train_Stats = {}
    Validation_Stats = {}
    Models = {}
    kernel = "NN"
    MaxNumHiddenNeurons = int(2*train_data.shape[1])+1
    MaxNumEpochs = 250
    LearningRates = [0.1,0.05,0.005]

    for hd in range(train_data.shape[1]+1,MaxNumHiddenNeurons,1):
        for numEpochs in range(100,MaxNumEpochs,100):
            for lr in LearningRates:
                for linNeuron in [True,False]:                   
                            
                    neural_net = neuralnet.SimpleNeuralNet(train_data.shape[1],num_hidden_neurons=hd, 
                                                           num_epochs=numEpochs,LearningRate=lr,include_LinearNeuron = linNeuron,
                                                           include_InputBias=True,include_OutputBias=True)
                    net = neural_net.train(train_data,train_target,plot=False)
                    print "weights_HO:",net.weights_HO
                    print "weights_HI:",net.weights_IH
                    
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


    write_stats_2file(train_fn,Models,Train_Stats,Validation_Stats,train_data,validation_data)  

