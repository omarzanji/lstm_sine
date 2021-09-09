import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation

import matplotlib.pyplot as plt

def generate_data(step_size=30):
    x_line = np.arange(-50*np.pi, 50*np.pi, 0.1)
    data = np.sin(x_line)
    step_size = 30
    x = []
    y = []
    end_ndx = 0
    for ndx,val in enumerate(x_line):
        seq = x_line[ndx : ndx+step_size]
        end_ndx += step_size
        try:
            next_point = x_line[end_ndx]
        except IndexError:
            x.append(seq)
            y.append(next_point)
            break
        x.append(seq)
        y.append(next_point)
    return np.array(x),np.array(y)


class LSTM_model:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(30, input_shape=(30, 1)))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model):
        hist = model.fit(self.x, self.y, epochs=20, verbose=1)
        model.summary()
        return hist


if __name__=="__main__":
    x_train, y_train = generate_data(30)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    sin_gen = LSTM_model(x_train, y_train)
    sin_model = sin_gen.create_model()
    history = sin_gen.train_model(sin_model)
    plt.plot(history.history['loss'], label="loss")
    plt.legend(loc="upper right")
    plt.show()