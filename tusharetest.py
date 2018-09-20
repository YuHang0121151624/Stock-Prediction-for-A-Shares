# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:01:46 2018
@author: Administrator
"""

import tushare as ts
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import traceback

VISUAL_TRUE_COLOR = 'red'
VISUAL_PRED_COLOR = 'green'
VISUAL_ALPHA = 0.7

# stimu function
def atan(x): 
    return tf.atan(x)

#visualize func
def Visualize(true, pred):
    try:
        #visualize the results
        plt.title('Return Rate Estimation')
        xindex = range(len(true))
        true = true/10
        plt.plot(xindex, true, color = VISUAL_TRUE_COLOR, alpha=VISUAL_ALPHA, label='true return rate')
        plt.plot(xindex, pred, color = VISUAL_PRED_COLOR, alpha=VISUAL_ALPHA, label='estimation')
        plt.xlabel('Date')
        plt.ylabel('Return Rate')
        plt.legend()
        plt.show()
    except:
        print (traceback.print_exc())

def PredictWithRNN(opt, lstm_inshape, lstm_units, lstm_act, lstm_dropw, lstm_dropu):
    try:
        #create model by keras
        lstm_input = Input(shape=lstm_inshape, name='lstm_input')
        #set hyper parameters
        lstm_output = LSTM(lstm_units, activation=atan, dropout_W=lstm_dropw, dropout_U=lstm_dropu)(lstm_input)
        Dense_output_1 = Dense(64, activation='linear', kernel_regularizer=keras.regularizers.l1(0.))(lstm_output)
        Dense_output_2 = Dense(16, activation='linear')(Dense_output_1)
        predictions = Dense(1, activation=lstm_act)(Dense_output_2)
        model = Model(input=lstm_input, output=predictions)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.fit(train_x, train_y, batch_size=conf.batch, nb_epoch=10, verbose=2)
        #make predictions
        pred = model.predict(test_x)
        return pred
    except:
        print (traceback.print_exc())

# basic config
class conf:
    instrument = '600000' #code of shares
    start_date = '2005-01-01'
    split_date = '2015-01-01' #before split: training data
    end_date = '2015-12-01' #after split: test data
    fields = ['close', 'open', 'high', 'low', 'volume', 'amount'] #amount = close*volume
    seq_len = 30 #each sample
    batch = 100 #one gradient descent with 100 samples 

# geting data from tushare and preprocessing

df = ts.get_k_data(conf.instrument, conf.start_date, conf.end_date)
#df.to_csv("600000data.csv", index=False, sep=',')
df['amount'] = df['close']*df['volume']
df['return'] = df['close'].shift(-5) / df['close'].shift(-1) - 1 #return(yield rate) = close price 5 days after / close price tomorrow
#df['return'] = df['return'].apply(lambda x:np.where(x>=0.2,0.2,np.where(x>-0.2,x,-0.2)))
df['return'] = df['return']*10 #just for training
df.dropna(inplace=True)
dftime = df['date'][df.date>=conf.split_date]
df.reset_index(drop=True, inplace=True)
scaledf = df[conf.fields]
traindf = df[df.date<conf.split_date]

#make the dataset
train_input = []
train_output = []
test_input = []
test_output = []
for i in range(conf.seq_len-1, len(traindf)):
    a = scale(scaledf[i+1-conf.seq_len:i+1])
    train_input.append(a)
    c = df['return'][i]
    train_output.append(c)
    
for i in range(len(traindf), len(df)):
    a = scale(scaledf[i+1-conf.seq_len:i+1])
    test_input.append(a)
    c = df['return'][i]
    test_output.append(c)

train_x = np.array(train_input)
train_y = np.array(train_output)
test_x = np.array(test_input) 
test_y = np.array(test_output)

#run the model
pred_y = PredictWithRNN('adam', (30,6), 128, atan, 0.2, 0.1)
#call visualize
Visualize(test_y, pred_y)
