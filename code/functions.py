# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:36:03 2014

@author: ogavril

PURPOSE: functions used by multiple programs and other ***_function.py

OUTPUT:  none 
"""

import numpy as np
import math
import pandas as pd
import statsmodels.api as sm

##### data prep functions ########################

def transform_categorical_vars(DF,categorical_vars):
    for var in categorical_vars:
        new_vars = sm.tools.categorical(np.array(DF[var]),drop=True)
        for i in range(new_vars.shape[1]):
            DF[var+':'+str(i)] = pd.Series(new_vars[:,i],index = DF.index)
    return DF
    

def def_cross_validation_subsets(df,varN,numK=5):
    df[varN] = -1
    for i in xrange(len(df.index)):
        df[varN].iloc[i] = i%numK
    return df

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
    
def silly_cuberoot(col):
    col1 = []
    for elem in col:
        if elem < 0:
            col1.append(-1*math.pow(-elem,1./3.))
        else:
            col1.append(math.pow(elem,1./3.))
    col1 = np.array(col1)
    return col1

def MSE(predicted_vals,true_vals):
    a = predicted_vals - true_vals
    mse = np.dot(a,a)/float(len(a))
    return mse
    
    
def AbsError(predicted_vals,true_vals):
    a = np.abs(predicted_vals - true_vals)
    return np.mean(a)

def determine_correlation(var1,var2):
    """assumes NaNs have been dropped"""
    v1 = np.array(var1)
    v2 = np.array(var2)
    mat = np.c_[(v1,v2)]# np.vstack((v1,v2)) #
    corr = np.corrcoef(mat.T)
    return corr[0][1]
    
def normalize(vec):
    """i think there is an sp.linalg.norm function, but for some reason it's not working for me possibly because 
    I don't require that vec is an np.array """
    min_ = np.min(vec)
    max_ = np.max(vec)
    if min_ != max_:
        n_vec = (vec-min_)/(max_-min_)
        return n_vec

    return vec

def scale_npvec(np_vec):
    tmp = []
    min_ = np.min(np_vec)
    max_ = np.max(np_vec)
    if min_ != max_:
        for i in range(len(np_vec)):
            tmp.append((np_vec[i] - min_)/(max_ - min_)*2 - 1)
        return tmp
    else:
        return np.ones(len(np_vec))
        
def randomize_prediction_v1(df,target_name):
    vals = [elem for elem in df[target_name].unique()]
    probs =[0] +[0 for x in range(len(vals))]
    for v in range(len(vals)):
        count = df[target_name][df[target_name] == vals[v]].count()
        probs[v+1] = probs[v]+ float(count)/float(len(df[target_name]))
    
    print "'success' probabilities:", probs    
    random_target = np.array(np.zeros(len(df[target_name])))
    t_ = np.random.uniform(low=0.0,high=1.0,size=len(df[target_name]))
    for i in xrange(len(df[target_name])):
        for v in xrange(1,len(probs)):
            if probs[v-1] <= t_[i] and t_[i] <= probs[v]:
                random_target[i] = vals[v-1]
                break
    return random_target

def calculate_SensSpecifPrecAccurNN(target_predicted,target):
    
    if len(target_predicted) == len(target):
        numTP = len(target_predicted[(target_predicted==1) & (target==1)])
        numFP = len(target_predicted[(target_predicted==1) & (target==-1)])
        numTN = len(target_predicted[(target_predicted==-1) & (target==-1)])
        numFN = len(target_predicted[(target_predicted==1) & (target==1)])
        sensitivity = float(numTP)/float(max(numTP+numFN,1))
        specificity = float(numTN)/float(max(numTN+numFP,1))
        precision = float(numTP)/float(max(numTP+numFP,1))
        accuracy = float(numTP +numTN)/float(numTP +numTN +numFP +numFN)
        return sensitivity,specificity,precision,accuracy
    else:
        return None

def cal_TP_FP_FN_TN_NN(target_predicted,target):
    if len(target_predicted) == len(target):
        numTP = len(target_predicted[(target_predicted==1) & (target==1)])
        numFP = len(target_predicted[(target_predicted==1) & (target==-1)])
        numTN = len(target_predicted[(target_predicted==-1) & (target==-1)])
        numFN = len(target_predicted[(target_predicted==-1) & (target==1)])
    return numTP,numFP,numFN,numTN,

