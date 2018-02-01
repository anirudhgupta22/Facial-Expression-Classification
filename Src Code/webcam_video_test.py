# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:47:47 2016

@author: Anirudh
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
print ('Model Loaded....')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('my_model_221.h5')
print ('Weights Loaded....')

########################################



##################WEBCAM DATA


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
ans = []
time = []
t = 0
p = 0
i = 0
c = 0
x_test = np.empty([1,1,48,48])
while(True):
    ret,frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame;
    cv2.imshow('img',frame)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print "Found {0} faces!".format(len(faces))
    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        i1 = frame[y-p:y+h+p,x-p:x+w+p]

        pts1 = np.float32([[x-p,y-p],[x+w+p,y-p],[x-p,y+p+h],[x+w+p,y+h+p]])
        pts2 = np.float32([[0,0],[255,0],[0,255], [255,255]])
        
        
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img = cv2.warpPerspective(frame,M,(256,256))
        input_img = cv2.resize(img, (48,48))
        
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = np.array(gray_image, dtype='float32')
        input_img = input_img / 255
        
        x_test[0,0,:,:] = input_img
        
        title = 'face'
        cv2.imshow(title,img)
        


        c = c + 1
        i = i + 1
        
        
        ############# MODEL ##########################
        if(c % 5 == 0):
            score =  model.predict(x_test, verbose = 0)
        
            print c
           
            exp, prob = find_exp(score)
            print(exp)
            print(prob)

###############Graph
            y_pos = np.arange(len(expressions))
                
            f = plt.figure(2)
            plt.ion()
            plt.clf()
            plt.bar(y_pos, score[0,:], align='center', alpha=0.5, color='blue')
            plt.xticks(y_pos, expressions)
            plt.ylabel('Probability')
            plt.title('Facial Experssion Prediction(Instant)')
             
            f.show()
            t = t + 1
            g = plt.figure(3)
            ans.insert(len(ans),expressions.index(exp[0]))
            time.insert(len(time),t)
            plt.ion()
            plt.clf()
           
            plt.plot(time[-120:], ans[-120:])
            plt.yticks(y_pos, expressions)
            plt.ylabel('Probability')
            plt.title('Facial Experssion Prediction(Overall)')

            g.show()

        
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
