# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:55:18 2014

@author: ogavril
PURPOSE: to create data which later can be used for building models;
         I used it mostly to develop and test the neuralnet module
"""


import numpy as np
import os
import math
# Create train samples

def write2file(data,target,fn=None):
    if fn == None:
        fn = "some_data.csv"
    fn = open(fn,'w')
  
    for c in range(data.shape[1]):
        fn.write("var_"+str(c)+",")
    fn.write("response\n")

    for r in range(data.shape[0]):
        l = ""
        for c in range(data.shape[1]):
            l += str(data[r][c])+","
        l += str(target[r][0])
        fn.write(l+"\n")
    fn.close()

        
        
def simple_addition():
    print "a simple neural network based on addition"
    size = 1000
    vec1 = np.random.uniform(-0.5, 0.5,size)
    vec2 = np.random.uniform(-0.5, 0.5,size)
    #print input
    #target = (input[:, 0] + input[:, 1]).reshape(size, 1)
    target = vec1+vec2
    target = target.reshape(size,1)
    inp = np.c_[vec1,vec2]
    write2file(inp,target,fn="test01.csv")
    
def NN_simple1():
    print "a neural network based on "
    size = 1000
#    vec1 = np.random.uniform(1,10,size)
#    vec2 = np.random.uniform(-1,1,size)
#    vec3 = np.random.uniform(-5,5,size)
    vec1 = np.random.uniform(-1,1,size)
    vec2 = np.random.uniform(-1,1,size)
    vec3 = np.random.uniform(-1,1,size)

    tvec1 = 2.5*(vec1)
    tvec2 = vec2 - np.ones(len(vec2))
    target = tvec1+tvec2+vec3
    #target = vec1+vec2+vec3
    target = target.reshape(size,1)
    inp = np.c_[vec1,vec2,vec3]
    write2file(inp,target,fn="test04.csv")


def test05_tanh():
    print "a neural network based on "
    size = 1000
    vec1 = np.random.uniform(-5,5,size)
    vec2 = np.random.uniform(-5,5,size)
    vec3 = np.random.uniform(-5,5,size)
    
    #print "orig data:\n",np.c_[vec1,vec2,vec2]
    target = np.tanh(vec1+vec2+vec3)
    target = target.reshape(size,1)
    inp = np.c_[vec1,vec2,vec3]
    write2file(inp,target,fn="test05.csv")

def data_set3():
    size = 1000
    vec1 = np.linspace(-10,10,size)
    vec2 = np.linspace(-1,1,size)
    vec3 = np.linspace(-5,5,size)
    
    tvec1 = vec1 + np.random.uniform(-0.05,0.05,size)
    target = np.c_[tvec1+vec2+vec3].reshape(size,1)
    write2file(np.c_[vec1,vec2,vec3],target,fn="test03.csv")
    
      

def NN_log_ff():
    print "a neural network based on addition of log and exp"
    size = 1000
    vec1 = np.random.uniform(1,10,size)
    vec2 = np.random.uniform(-1,1,size)
    vec3 = np.random.uniform(-5,5,size)

    tvec1 = np.log(vec1)
    tvec2 = np.exp2(vec2)
    target = np.c_[tvec1+tvec2+vec3].reshape(size,1)
    write2file(np.c_[vec1,vec2,vec3],target,fn="test02_logNexp.csv")

def simple_square():
    print "square and simple addition"
    size = 1000
    vec = np.random.uniform(-5,5,size)
    vec2 = np.random.uniform(-0.5,0.5,size)

    tvec = np.array([x**2 for x in vec])
    #target = np.c_[tvec+vec].reshape(size,1) 
    target = np.c_[tvec+vec2].reshape(size,1) 

    write2file(np.c_[vec,vec2],target,fn="test06_Square_plus_random.csv") #"test07_SquarePlus.csv")
    
    
if __name__=="__main__":
    #simple_addition()
    #NN_log_ff()
    #NN_simple1()
    #test05_tanh()
    simple_square()
    #data_set3()
