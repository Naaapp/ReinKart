


import argparse


import sys
sys.path.append("game/")
import numpy as np
from collections import deque

from PIL import Image

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

import matplotlib.pyplot as plt



LEARNING_RATE = 1e-3
N_INPUT = 3
N_OUTPUT = 3
img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 3 #RGB


def create_model(keep_prob=0.6):
    model = Sequential()

    print("Now we build the model")
    model = Sequential()
    model.add(Dense(output_dim=120, activation='relu', input_dim=N_INPUT))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=N_OUTPUT, activation = 'softmax'))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

    

    
        