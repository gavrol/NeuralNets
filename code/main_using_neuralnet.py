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

def def_cross_validation_subsets(df,varN,numK=5):
    df[varN] = -1
    for i in xrange(len(df.index)):
        df[varN].iloc[i] = i%numK
    return df
    
    
    
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
    test_set = validation_set +1
    if test_fn == None:
        #if there is no test file, then make a test data set out of train, but using a bit of it
        train_df = def_cross_validation_subsets(train_df,train_var,numK=validation_set+2)
        test_df = train_df[train_df[train_var] == test_set]
        train_df = train_df[train_df[train_var] != test_set]
    else:
        test_df = initialize_test_data(DATA_DIR,test_fn)
        train_df = def_cross_validation_subsets(train_df,train_var,numK=validation_set+1)
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
    
    test_data = make_numpy_matrix(test_df,explanatory_vars)
    test_target = np.array(test_df[target_var])#.reshape(test_data.shape[0],1)
                
    
    #neural_net.test(test_data)
    TrainModel_Stats = {}
    TestModel_Stats = {}
    kernel = "NN"
    neural_net = neuralnet.SimpleNeuralNet(train_data.shape[1],num_hidden_neurons=train_data.shape[1]+2, 
                                           num_epochs=200,LearningRate=0.07,include_LinearNeuron = True,
                                           include_InputBias=True,include_OutputBias=True)
    net = neural_net.train(train_data,train_target)
    print "weights_HO:",net.weights_HO
    print "weights_HI:",net.weights_IH
    
    predicted_values_train = neural_net.validate(train_data,train_target) 
    predicted_values_validation = neural_net.validate(validation_data,validation_target)
    tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
    TrainModel_Stats[kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
    ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = functions.calculate_SensSpecifPrecAccur(predicted_values_validation,validation_target)                
    TestModel_Stats[kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}  
    s =  kernel+"\n"
    s += "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
    s += "For the validation set "+str(validation_set)+" of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)

    print s
   
