#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:23:43 2018

@author: kiyoshitaro
"""

import numpy as np
import pandas as pd
import datetime
import os 
import sklearn 
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import math

valid_set_size_percentage = 10 
test_set_size_percentage = 10 
#print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
#print(os.getcwd()+':', os.listdir(os.getcwd()) , "\n\n" , os.supports_fd);
#print(os.path.dirname('/home/kiyoshitaro/Desktop/ML/randomfaces4ar'))
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)
df.info()
df.head()
df.describe()
#print(list(set(df.symbol))[:15], "\n", df.tail())

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(df[df.symbol == 'EQIX'].open.values,color = 'red', label = 'open' )
#values to ---> numpy
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [day]')
plt.ylabel('volumn')
plt.legend(loc = 'best')

plt.subplot(1,2,2);
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');

