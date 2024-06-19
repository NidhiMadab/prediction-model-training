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


def model(train_X, train_y, val_X, val_y):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides = 1, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides = 1))
    model.add(MaxPooling1D(pool_size=2, strides = 2))
    model.add(LSTM({{choice([8, 16, 32, 64, 128])}},
                   activation='tanh',
                   return_sequences=False))
    model.add(Flatten())
    model.add(Dense({{choice([8, 16, 32, 64, 128])}}, activation='relu'))
    model.add(Dense({{choice([8, 16, 32, 64, 128])}}, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_X, train_y,
              batch_size={{choice([512, 1024, 2048, 4096])}},
              epochs={{choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])}},
              verbose=1,
              validation_data=(val_X, val_y))
    loss = model.evaluate(val_X, val_y, verbose=1)
    print('Test loss:', loss)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


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

start = time.time()
best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=300,
                                      trials=Trials())
end = time.time()

print((end-start)/60)
save_model(best_model, "cnn")
print(best_run)
print(best_model.summary())
