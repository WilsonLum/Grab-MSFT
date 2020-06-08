#######################################################
# Model definitions
# 08/06/2020 - Wilson Lum
#
#######################################################
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from math import floor


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras import optimizers, regularizers

# defining global variables

# ----------------------------------------------------------------------------
# Define the deep learning models
# ----------------------------------------------------------------------------
def createModel(X_train, predict_next_no_of_output, index = 0):

    # For LSTM - Use indexes 0 to 10
    if   (index <= 0): return LSTM1_Model(X_train, predict_next_no_of_output)
    elif (index == 1): return LSTM2_Model(X_train, predict_next_no_of_output)
    # For Bi-LSTM - Use indexes 11 to 20
    elif (index == 11): return BiLSTM1_Model(X_train, predict_next_no_of_output)
    elif (index == 12): return BiLSTM2_Model(X_train, predict_next_no_of_output)
    #
    # For CNN - Use indexes 21 to 30
    elif (index == 21): return None
    #
    else: return None

def LSTM1_Model(X_train,predict_next_no_of_output): 
    inputs  = Input(shape=(X_train.shape[1],X_train.shape[2]))
    y = LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    y = BatchNormalization()(y)
    y = LSTM(1024, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(y)
    y = BatchNormalization()(y)
    y = LSTM(1024, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(y)
    y = BatchNormalization()(y)
    y = LSTM(512, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)(y)
    y = BatchNormalization()(y)
    y = LSTM(512, return_sequences=True, dropout=0.5,recurrent_dropout=0.5)(y)
    y = BatchNormalization()(y)
    y = LSTM(128, dropout=0.5,recurrent_dropout=0.5)(y)
    y = BatchNormalization()(y)
    y = Dense(predict_next_no_of_output, activation='sigmoid')(y)
  
    model = Model(inputs=inputs,outputs=y)
    model.compile(loss='mae',optimizer='adam', metrics=['mse', 'mae'])
    return model

def LSTM2_Model(X_train,predict_next_no_of_output): 
    inputs  = Input(shape=(X_train.shape[1],X_train.shape[2]))
    y = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    y = BatchNormalization()(y)
    y = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(y)
    y = BatchNormalization()(y)
    y = LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)(y)
    y = BatchNormalization()(y)
    y = LSTM(128, dropout=0.5,recurrent_dropout=0.5)(y)
    y = BatchNormalization()(y)
    y = Dense(predict_next_no_of_output, activation='sigmoid')(y)
  
    model = Model(inputs=inputs,outputs=y)
    model.compile(loss='mae',optimizer='adam', metrics=['mse', 'mae'])
    return model

def BiLSTM1_Model(X_train,predict_next_no_of_output): 
    inputs  = Input(shape=(X_train.shape[1],X_train.shape[2]))
    y = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=1024, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=1024, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=512, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=512, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5))(y)
    y = BatchNormalization()(y)
    y = Dense(predict_next_no_of_output, activation='sigmoid')(y)
  
    model = Model(inputs=inputs,outputs=y)
    model.compile(loss='mse',optimizer='adam', metrics=['mse', 'mae'])
    return model

def BiLSTM2_Model(X_train,predict_next_no_of_output): 
    inputs  = Input(shape=(X_train.shape[1],X_train.shape[2]))
    y = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(y)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5))(y)
    y = BatchNormalization()(y)
    y = Dense(predict_next_no_of_output, activation='sigmoid')(y)
  
    model = Model(inputs=inputs,outputs=y)
    model.compile(loss='mse',optimizer='adam', metrics=['mse', 'mae'])
    return model
