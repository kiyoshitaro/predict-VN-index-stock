#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:02:50 2018ake_blo

@author: kiyoshitaro
"""
import time
from sklearn.datasets import make_blobs 
from keras.datasets import imdb,fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_data(pl,X,y):
    pl.plot(X[y==0,0],X[y==0,1],'ob',alpha = 0.5)
    pl.plot(X[y==1,0],X[y==1,1],'xr',alpha = 0.5)
    pl.legend(['0','1'])
    return pl

def plot_decision_boundary(model , X , y):
    amin, bmin  = X.min(axis= 0) -0.1
    amax , bmax = X.max(axis = 0) + 0.1
    hticks = np.linspace(amin, amax,101)
    vticks = np.linspace(bmin , bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c = model.predict(ab)
    Z = c.reshape(aa.shape)
    plt.figure(figsize= (12,8))
    plt.contourf(aa,bb,Z,cmap = 'bwr',alpha = 0.2)
    
    plot_data(plt,X,y)
    return plt

start_time = time.time()

X,y = make_blobs(n_samples = 100, centers = 2, random_state=42)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

pl = plot_data(plt,X,y)
pl.show()
elapsed_time = time.time() - start_time
print("\n Time : " , elapsed_time )


from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

#import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam 

start_time = time.time()

model = Sequential()

model.add(Dense(1,input_shape = (2,) , activation = "sigmoid"))
model.compile(Adam(lr = 0.05), 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train,epochs= 100, verbose = 0)
eval_result = model.evaluate(X_test, y_test)
elapsed_time = time.time() - start_time
print("\n\n Test lost : ", eval_result[0] , "test acccuracy" , eval_result[1], "\n Time : " , elapsed_time )







