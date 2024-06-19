import random
import pandas as pd
import numpy as np
from datetime import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import save_model

random.seed(1234)


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
model.add(Dense(128,
               input_dim=(train_X.shape[1]),
               activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(loss='mse',optimizer="adam")
model.fit(train_X, train_y,
          batch_size=2048,
          epochs=30,
          verbose=1,
          validation_data=(val_X, val_y))
loss = model.evaluate(val_X, val_y, verbose=1)


save_model(model, "mlp")
