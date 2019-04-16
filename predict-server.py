import sys, time, logging, os, argparse

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
from skimage.io import imread
from skimage.io import imsave

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

import numpy as np
from PIL import Image, ImageGrab
from socketserver import TCPServer, StreamRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from train import create_model, is_valid_track_code, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS

from collections import deque
from keras.optimizers import SGD , Adam

OUT_SHAPE = 1

INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3

VALIDATION_SPLIT = 0.1
USE_REVERSE_IMAGES = False


ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 36. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.01 # starting value of epsilon
REPLAY_MEMORY = 100 # number of previous transitions to remember
BATCH = 4 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

LEARNING_RATE = 1e-4

model = create_model(keep_prob=1)

#print ("Now we load weight")
#model.load_weights("model12.h5")
#adam = Adam(lr=LEARNING_RATE)
#model.compile(loss='mse',optimizer=adam)
#print ("Weight load successfully")    


def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr


def train_network( s_t, D,action_index, r_t, init, t, distance, cos, sin, velocity):
    
    if t == OBSERVATION:
        print("observation finished")
    

    loss = 0
    # get the first state by doing nothing and preprocess the image to 80x80x4
    if t == 1:

        x_t = [distance, cos, sin, velocity]
        
        #s_t = np.stack((x_t, x_t, x_t, x_t))
        
        #if we not stack
        s_t = np.array([x_t])
        

        #In Keras, need to reshape
#        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
        action_index = 1
        
        returnvalue = [0,1,0]
        return returnvalue, s_t, D, action_index #go straight forward at init, no need to use the model
        
    
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    
    x_t1 = [distance, cos, sin, velocity]

    #s_t1 = np.stack((x_t1, s_t[0], s_t[1], s_t[2]))
    
    #if we not stack
    s_t1 = np.array([x_t1])

    
    # store the transition in D
    terminal = 1
    D.append((s_t, action_index, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    #only train if done observing
    if t > OBSERVE:

        #sample a minibatch to train on
        minibatch = random.sample(D, BATCH)

        #Now we do the experience replay
        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        
        print(state_t)

        targets = model.predict(state_t)
        Q_sa = model.predict(state_t1)
        targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

        loss += model.train_on_batch(state_t, targets)

    s_t = s_t1
    t = t + 1
    
    if t % 20 == 0:
        print("Now we save model")
        model.save_weights("model13.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)
    
    a_t = np.zeros([ACTIONS])
    #choose an action epsilon greedy
    if t % FRAME_PER_ACTION == 0:
        if t < OBSERVE:
            action_index = 1
            a_t[action_index] = 1
        elif random.random() <= epsilon:            
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
#            print("----------Predicted Action----------")
            
            print(s_t)
#            print(s_t)
            q = model.predict(s_t)       #input a stack of 4 images, get the prediction

            max_Q = np.argmax(q)
            print(max_Q)
            action_index = max_Q
            a_t[max_Q] = 1

    #We reduced the epsilon gradually
#    if epsilon > FINAL_EPSILON and t > OBSERVE:
#        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
    return a_t, s_t, D, action_index



class TCPHandler(StreamRequestHandler):
    def handle(self):

        logger.info("Handling a new connection...")
        t = 0

        D = deque()
        s_t = 0
        action_index = 0
        
        for line in self.rfile:
            t = t + 1
            message = str(line.strip(),'utf-8')
            logger.debug(message)

            if message.startswith("MESSAGE"):
                index_score = message.find("SCORE")
                index_distance = message.find("DISTANCE")
                index_cos = message.find("COS")
                index_sin = message.find("SIN")
                index_velocity = message.find("VELOCITY")
                
                init = message[7]
                
                index_begin = index_score + 5
                index_end = index_distance - 1
                score = float(message[index_begin:index_end]) - 1
                index_begin = index_distance + 8
                index_end = index_cos - 1
                distance = float(message[index_begin:index_end])
                index_begin = index_cos + 3
                index_end = index_sin - 1
                cos = float(message[index_begin:index_end]) 
                index_begin = index_sin + 3
                index_end = index_velocity - 1
                sin = float(message[index_begin:index_end]) 
                index_begin = index_velocity + 8
                velocity = float(message[index_begin:index_begin+10]) 

                prediction, s_t, D, action_index = train_network(s_t, D, action_index, score, init, t, distance, cos, sin, velocity)
                self.wfile.write((str(int(prediction[0])) + (str(int(prediction[1]))) + (str(int(prediction[2]))) + "\n").encode('utf-8'))


            if message.startswith("PREDICT:"):
                im = Image.open(message[9:])
                prediction = model.predict(prepare_image(im), batch_size=1)[0]
                self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('-a', '--all', action='store_true', help='Use the combined weights for all tracks, rather than selecting the weights file based off of the course code sent by the Play.lua script.', default=False)
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading model...")


#    if args.all:
#        model.load_weights('weights/all.hdf5')

    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
