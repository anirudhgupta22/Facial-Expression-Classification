from __future__ import absolute_import
from __future__ import print_function
from load_data import faces_load_data
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import csv
from keras.optimizers import SGD
from keras.optimizers import Adam
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
import pandas
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras import backend as K

#############################################

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 50

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = model.predict(X_train)
            self.predictions.append(pred)



###############################################






np.random.seed(1337)  # for reproducibility
K.set_image_dim_ordering('th')
batch_size = 128
nb_classes = 7
nb_epoch = 100

img_rows, img_cols = 48, 48

nb_filters = 32

nb_pool = 2

nb_conv = 3

(X_train, Y_train), (X_test, Y_test) = faces_load_data()
print('hi')
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0],1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#print(X[1,:,:])
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train,7)
Y_test = np_utils.to_categorical(Y_test,7)

print (Y_train.shape)
num_classes = Y_train.shape[1]
print (num_classes)

# model architecture:
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='full', input_shape=(1, 48, 48), activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='full', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))


model.add(Convolution2D(64, 3, 3, border_mode='full', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='full', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(128, 3, 3, border_mode='full', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='full', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))



model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

history = TrainingHistory()
# optimizer:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save('model_221.h5')
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print ('Training....')
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=False, verbose=1, validation_data=(X_test, Y_test), callbacks=[history])

####################





#model.load_weights('my_model_weights_20_11.h5')
model.save_weights('my_model_221.h5')   
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(history.losses)
print(type(history))
print(shape(history.losses))

import csv
ofile  = open('loss.csv', "wb")
writer = csv.writer(ofile, delimiter=',')
x = [1,3,4]
writer.writerow(history.losses)
ofile.close()
