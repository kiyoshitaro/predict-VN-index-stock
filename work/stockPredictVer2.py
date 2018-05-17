#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 01:57:02 2018

@author: kiyoshitaro
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot
import math
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time



####### set to test #########
train_size_percentage = 0.8
####### set to test #########




# scale  to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0],
                          train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0],
                        test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#take hítory
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
#    return columns
    return df

#solve difference between two step 
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)    
    
    return pd.Series(diff)


# invert differenced 
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def invert_scale(scaler,X,value):
    new_row = [x for x in X] + [value]
    arr = np.array(new_row).reshape(1,len(new_row))
    invert = scaler.inverse_transform(arr)
    return invert[0,-1]
    
#build  lstm model
def fit_lstm(train, batch_size,nb_epoch, neurons ):
    X,y = train[:,0:-1], train[:,-1]
    X = X.reshape(X.shape[0],1,X.shape[1])
    
    #X=[ [[1,2,3,...]] , [[2,3,4,....]] , [[3,4,5,....]] , ....]
    #init
    model = Sequential()
    #choose neurons, batch (3-dimensions) , stateful
    model.add(LSTM(neurons, batch_input_shape = (batch_size, X.shape[1], X.shape[2]),stateful= True))
    model.add(Dense(1))
    
    
    
    ####### set to test #########
    model.compile(loss='mean_squared_error', optimizer='adam')
    ####### set to test #########
    
    
    

    for i in range(nb_epoch):
        #verbose 
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        
    return model

def forecast_lstm(model, batch_size, X):
    #reshape X to fit keras
    X = X.reshape(1, 1, len(X))  
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

start = time.time()
series = pd.read_csv('raw_data.csv', 
                  header = None,
                  parse_dates = [0],
                  index_col = 0,
                  squeeze = True, #to be series
                  )
pyplot.plot(series,label =  "close_price",color = 'green')
pyplot.title("stock price")
pyplot.xlabel('time')
pyplot.ylabel('close_price')
pyplot.show()

datas = series.values ##to numpy
differenced = difference(datas, 1)

supervised = timeseries_to_supervised(differenced,1)
#to numpy
supervised_values = supervised.values


print(differenced.head())

#devide train/test
train_size = int(np.round(train_size_percentage*datas.size))
test_size = len(series)-1-  train_size
train, test = supervised_values[0:train_size], supervised_values[train_size:]
scaler, train_scaled, test_scaled = scale(train, test)
 
# fit the model


batch_size = 1
nb_epoch = 10
neurons = 4




repeats = 20
error_scores = list()

for r in range(repeats):
    
        
        ####### set to test #########
    lstm_model = fit_lstm(train_scaled, batch_size,nb_epoch, neurons)
    ####### set to test #########
    
    
    
    
    # forecast the training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    
    

    
    ####### set to test #########
    lstm_model.predict(train_reshaped, batch_size=1)
    ####### set to test #########
    
    
    
    
    
    
    
    #predict for test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    
        yhat = forecast_lstm(lstm_model, 1, X)
    #    yhat= y
    
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(datas, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        expected = datas[len(train) + i + 1]
#        print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
     
        
        
    # report performance
#    rmse = math.sqrt(mean_squared_error(datas[-test_size:], predictions))
#    print('Test RMSE: %.3f' % rmse)


# line plot of observed vs predicted
#pyplot.plot(datas[-test_size:])


    rmse = math.sqrt(mean_squared_error(datas[-test_size:], predictions))
    print('%d) Test RMSE: %.3f' % (r+1, rmse))
    error_scores.append(rmse)
 
# summarize results
results = pd.DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()


end = time.time()
elapsed = end - start
#show data to compare result (a part not full)
#pyplot.plot(datas[-50:])
#pyplot.plot(predictions[-50:])
#pyplot.show()

print("With batch_size = {} ,nb_epoch = {} , neurons = {} in time = {} , Test RMSE: {}".format(batch_size,nb_epoch, neurons,elapsed,rmse))
#epoch=1000_rmse=

########################## Test Libarary and code  ##################################


#inverted = list()
#for i in range(len(differenced)):
#    value = inverse_difference(series, differenced[i], len(series)-i)
#    inverted.append(value)
#inverted = pd.Series(inverted)
#print(inverted.head())




#datas = datas.reshape(len(datas),1)
#scaler = MinMaxScaler(feature_range = (-1,1))
#scaler =scaler.fit(datas)
#scaled_X = scaler.transform(datas)
#scaled_series = pd.Series(scaled_X[:,0])


#print(scaled_series.head())
#
#inverted_X = scaler.inverse_transform(scaled_X)
#inverted_series = pd.Series(inverted_X[:, 0])
#print(inverted_series.head()) 


#
#history = [x for x in train ]
#predictions = list()
#for i in range(test_size):
## make prediction
#	predictions.append(history[-1])
#	# observation
#	history.append(test[i])
#    
#from sklearn.metrics import mean_squared_error
#rmse = math.sqrt(mean_squared_error(test, predictions))
#print('RMSE: %.3f' % rmse)
#
#pyplot.plot(test, label = "test", color = 'red')
#pyplot.plot(predictions , label = "predictions", color = 'blue')
#
#pyplot.show()



