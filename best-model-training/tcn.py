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



def data():
    # load train and val data
    train_X = np.load("train_X.npy")
    train_y = np.load("train_y.npy")
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1],1)
    train_y = train_y.reshape(train_y.shape[0])

    val_X = np.load("val_X.npy")
    val_y = np.load("val_y.npy")

    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1],1)
    val_y = val_y.reshape(val_y.shape[0])

    return train_X, train_y, val_X, val_y


train_X, train_y, val_X, val_y = data()


model = Sequential()
for i in range(6):
    dilation_rate = 2 ** i
    model.add(Conv1D(filters=3, kernel_size=5, padding='causal',
                     dilation_rate=dilation_rate, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1,activation='relu'))

model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_y,
          batch_size=2048,
          epochs=100,
          verbose=1,
          validation_data=(val_X, val_y))


save_model(model, "tcn")
