import pandas as pd
import numpy as np
from datetime import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import TimeDistributed

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    # load train and val data
    train_X = np.load("train_X.npy")
    train_y = np.load("train_y.npy")
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1],1)
    train_y = train_y.reshape(train_y.shape[0],1)

    val_X = np.load("val_X.npy")
    val_y = np.load("val_y.npy")
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1],1)
    val_y = val_y.reshape(val_y.shape[0],1)

    return train_X, train_y, val_X, val_y



model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides = 1, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides = 1))
model.add(MaxPooling1D(pool_size=2, strides = 2))
model.add(LSTM(128,
               activation='tanh',
               return_sequences=False))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


model.compile(loss='mse',optimizer="adam")
model.fit(train_X, train_y,
          batch_size=2048,
          epochs=30,
          verbose=1,
          validation_data=(val_X, val_y))


save_model(best_model, "cnn")
