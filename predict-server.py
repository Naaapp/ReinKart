import sys, time, logging, os, argparse
import matplotlib.pyplot as plt
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

from train import create_model
from collections import deque
from keras.optimizers import SGD , Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


INPUT_WIDTH = 80
INPUT_HEIGHT = 80
INPUT_CHANNELS = 3



ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 15. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.05 # starting value of epsilon
REPLAY_MEMORY = 100 # number of previous transitions to remember
BATCH = 10 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 3 #We stack 4 frames

LEARNING_RATE = 1e-3


model = create_model(keep_prob=1)

# Use it if you want to load the corresponding models

#print ("Now we load weight")
#model.load_weights("model60.h5")
#adam = Adam(lr=LEARNING_RATE)
#model.compile(loss='mse',optimizer=adam)
#print ("Weight load successfully")    




def train_network( s_t, D,action_index, r_t, init, t, distance, cos, sin, velocity, vx, vy): 
    
    if t == OBSERVATION:
        print("observation finished")
    

    if t == 1:

        x_t = [distance, vx, vy]
        
        #s_t = np.stack((x_t, x_t, x_t, x_t))
        
        #if we not stack
        s_t = np.array([x_t])
        

        action_index = 1
        
        returnvalue = [1,0,0]
        return returnvalue, s_t, D, action_index #go straight forward at init, no need to use the model
        
        
    
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    
    
    x_t1 = [distance, vx, vy]

    #s_t1 = np.stack((x_t1, s_t[0], s_t[1], s_t[2]))
    
    #if we not stack
    s_t1 = np.array([x_t1])

    
    # store the transition in D
    terminal = 0
    D.append((s_t, action_index, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    #only train if done observing
    if t > OBSERVE:
        #sample a minibatch to train on
        minibatch = random.sample(D, BATCH)
        
        for state_t, action_t, reward_t, state_t1, terminal in minibatch:
            target = reward_t
            if not terminal:
                target = reward_t + GAMMA * np.amax(model.predict(state_t1)[0])
            target_f = model.predict(state_t)
            target_f[0][np.argmax(action_t)] = target
            model.fit(state_t, target_f, epochs=1, verbose=0)

    s_t = s_t1
    t = t + 1
    
    if t % 30 == 0:
        print("Now we save model")
        model.save_weights("model60.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)
    
    a_t = np.zeros([ACTIONS])
    #choose an action epsilon greedy
    if t % FRAME_PER_ACTION == 0:
        if distance > 0.52 and distance < 0.7:
            epsilon = 0.5
        elif t < OBSERVE:
            action_index = 0
            a_t[action_index] = 1
        elif random.random() <= epsilon:            
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
#            print("----------Predicted Action----------")
#            print(s_t)
            q = model.predict(s_t)       
            print(q)
            max_Q = np.argmax(q[0])
            action_index = max_Q
            a_t[max_Q] = 1
        epsilon = INITIAL_EPSILON

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
        
        distance_array = []
        
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
                index_vx = message.find("VX")
                index_vy = message.find("VY")
                
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
                index_end = index_vx - 1
                velocity = float(message[index_begin:index_end])
                index_begin = index_vx + 2
                index_end = index_vy - 1
                vx = float(message[index_begin:index_end])
                index_begin = index_vy + 2
                index_end = index_vy + 12
                vy = float(message[index_begin:index_end])

                distance_array.append(distance)
                
                prediction, s_t, D, action_index = train_network(s_t, D, action_index, score, init, t, distance, cos, sin, velocity, vx, vy)
                self.wfile.write((str(int(prediction[0])) + (str(int(prediction[1]))) + (str(int(prediction[2]))) + "\n").encode('utf-8'))
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('-a', '--all', action='store_true', help='Use the combined weights for all tracks, rather than selecting the weights file based off of the course code sent by the Play.lua script.', default=False)
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()
    
    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading model...")


    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
