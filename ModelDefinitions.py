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
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, LSTM, Bidirectional
from tensorflow.keras import optimizers, regularizers

# defining global variables

# ----------------------------------------------------------------------------
# Define the deep learning models
# ----------------------------------------------------------------------------
def createModel(X_train, predict_next_no_of_output, index = 0):

    # For LSTM - Use indexes 1 to 10
    if   (index == 1): return LSTM1_Model(X_train, predict_next_no_of_output)
    elif (index == 2): return LSTM2_Model(X_train, predict_next_no_of_output)
    # For Bi-LSTM - Use indexes 11 to 20
    elif (index == 11): return BiLSTM1_Model(X_train, predict_next_no_of_output)
    elif (index == 12): return BiLSTM2_Model(X_train, predict_next_no_of_output)
    #
    # For CNN - Use indexes 21 to 30
    elif (index == 21): return CNN1_Model(X_train, predict_next_no_of_output)
    elif (index == 22): return CNN2_Model(X_train, predict_next_no_of_output)
    elif (index == 23): return CNN3_Model(X_train, predict_next_no_of_output)
    # For CNN - Use indexes 31 to 40
    elif (index == 31): return CNN_LSTM1_Model(X_train, predict_next_no_of_output)
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

def CNN1_Model(X_train, predict_next_no_of_output):
    
  xin = Input(shape=(X_train.shape[1], X_train.shape[2]))
  
  x = Conv1D(filters=512, kernel_size=3, padding='same',kernel_regularizer=regularizers.l2(0.001))(xin)
  x = Activation('relu')(x)
  x = MaxPooling1D(1)(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(1)(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(1)(x)
  x = BatchNormalization()(x)

  x = Flatten()(x)
  x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)
  x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.4)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  x = Dense(predict_next_no_of_output,activation='sigmoid',kernel_initializer='he_normal')(x)

  model = Model(inputs=xin, outputs=x)
  model.compile(loss='mse', optimizer='adam' , metrics=['mse', 'mae'])
  
  return model

def CNN2_Model(X_train, predict_next_no_of_output):
    
  xin = Input(shape=(X_train.shape[1], X_train.shape[2]))
  
  x = Conv1D(filters=512, kernel_size=3, padding='same',kernel_regularizer=regularizers.l2(0.001))(xin)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Flatten()(x)
  x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)
  x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.4)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  x = Dense(predict_next_no_of_output,activation='sigmoid',kernel_initializer='he_normal')(x)

  model = Model(inputs=xin, outputs=x)
  model.compile(loss='mse', optimizer='adam' , metrics=['mse', 'mae'])
  
  return model

def CNN3_Model(X_train, predict_next_no_of_output):
    
  xin = Input(shape=(X_train.shape[1], X_train.shape[2]))
  
  x = Conv1D(filters=512, kernel_size=3, padding='same',kernel_regularizer=regularizers.l2(0.001))(xin)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv1D(filters=1024, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Dense(predict_next_no_of_output,activation='sigmoid',kernel_initializer='he_normal')(x)

  model = Model(inputs=xin, outputs=x)
  model.compile(loss='mse', optimizer='adam' , metrics=['mse', 'mae'])
  
  return model

def CNN_LSTM1_Model(X_train, predict_next_no_of_output): 
  inputs  = Input(shape=(X_train.shape[1],X_train.shape[2]))
  y = Conv1D(64, 2, activation='relu')(inputs)
  y = Dropout(0.25)(y)
  y = Conv1D(48, 2, activation='relu')(y)
  y = Dropout(0.25)(y)
  y = MaxPooling1D(1)(y)
  y = Conv1D(32, 1, activation='relu')(y)
  y = Dropout(0.25)(y)
  y = MaxPooling1D(1)(y)
  y = Conv1D(32, 1, activation='relu')(y)
  y = Dropout(0.5)(y)
  y = Conv1D(16, 1, activation='relu')(y)
  y = Dropout(0.5)(y)
  y = LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(y)
  y = LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(y)
  y = LSTM(48, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(y)
  y = LSTM(32, return_sequences=True, dropout=0.5,recurrent_dropout=0.5)(y)
  y = LSTM(16, return_sequences=True, dropout=0.5,recurrent_dropout=0.5)(y)
  y = LSTM(2)(y)
  y = Dense(predict_next_no_of_output, activation='sigmoid')(y)
  
  model = Model(inputs=inputs,outputs=y)
  model.compile(loss='mse',optimizer='adam', metrics=['mse', 'mae'])
  return model