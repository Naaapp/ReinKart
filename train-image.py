#from __future__ import print_function

import glob
import os
import hashlib
import time
import argparse
from mkdir_p import mkdir_p

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import argparse
import sys
sys.path.append("game/")
import random
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



ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-3

N_INPUT = 4
N_OUTPUT = 3

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 3 #RGB



def create_model(keep_prob=0.6):
    model = Sequential()

    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), input_shape=(img_rows,img_cols,img_channels), strides=(4, 4), padding="same"))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(3))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

    
