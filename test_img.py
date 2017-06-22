# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:02:12 2016

@author: Annu
"""

import time
import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.optimizers import Adam
import pandas
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model


expressions = "Angry", "Disgust","Fear", "Happy","Sad", "Surprise", "Neutral"

def find_exp(score) :
    yx = zip(score[0,:],expressions)
    yx.sort(reverse=True)
    y1 = np.empty([7], dtype='float32')
    x1 = [x for y,x in yx]
    y1 = [round(y,3) for y,x in yx]
    
    return x1,y1




################################
K.set_image_dim_ordering('th')

model= load_model('model_221.h5')
model.load_weights('my_model_221.h5')
x_test = np.empty([1,1,48,48])


########################################

test_img = cv2.imread('p1.png')
test_img = cv2.resize(test_img, (48,48))
gray_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
input_img1 = np.array(gray_image, dtype='float32')
input_img2 = input_img1 / 255
x_test[0,0,:,:] = input_img2

score =  model.predict(x_test, verbose = 1)

exp, prob = find_exp(score)
print(exp)
print(prob)

y_pos = np.arange(len(expressions))
 
plt.bar(y_pos, score[0,:], align='center', alpha=0.5, color='green')
plt.xticks(y_pos, expressions)
plt.ylabel('Probability')
plt.title('Results')
 
plt.show()


