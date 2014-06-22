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
    train_fn = "test07_SquarePlusItself.csv"# "test02_logNexp.csv"#"test01.csv"#"test06_Square_plus_random.csv"#"test03.csv"#"test04.csv"##"test03_random.csv"#"test05.csv"#""test07_Square.csv"# 
    
    #observations,target,df = initialize_data(DATA_DIR,train_fn)
    train_df = initialize_train_data(DATA_DIR,train_fn)

    target_var = "response"

    """organize data into train, test,and vaildate"""
    train_var = "_TRAIN"
    validation_set = 4
    test_set = -1#validation_set +1

    train_df = functions.def_cross_validation_subsets(train_df,train_var,numK=validation_set+1)
    test_df = train_df[train_df[train_var] == test_set]
    train_df = train_df[train_df[train_var] != test_set]
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
    

    hdn = train_data.shape[1]+2
    numEpochs = 200
    lr = 0.05
    linNeuron = True
                            
    neural_net = neuralnet.SimpleNeuralNet(train_data.shape[1],num_hidden_neurons=hdn, 
                                           num_epochs=numEpochs,LearningRate=lr,include_LinearNeuron = linNeuron,
                                           include_InputBias=True,include_OutputBias=True)
    net = neural_net.train(train_data,train_target,plot=True)
    print "weights_HO:",net.weights_HO
    print "weights_HI:",net.weights_IH
    
    predicted_values_train,RMSE_train = neural_net.validate(train_data,train_target) 
    predicted_values_validation,RMSE_validation= neural_net.validate(validation_data,validation_target)
                    
   
