from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D,PReLU
from keras import backend as K
import numpy as np
import sys
from keras.callbacks import *
import TYY_callbacks
from keras.optimizers import SGD, Adam


batch_size = 128
num_classes = 10
epochs = 30

isCenterloss = True
#isCenterloss = False



# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(28,28,1))
x = Conv2D(32, (5,5))(inputs)
x = PReLU()(x)
x = Conv2D(32, (5,5))(x)
x = PReLU()(x)
x = Conv2D(64, (3,3))(x)
x = PReLU()(x)
x = Conv2D(64, (5,5))(x)
x = PReLU()(x)
x = Conv2D(128, (5,5))(x)
x = PReLU()(x)
x = Conv2D(128, (5,5))(x)
x = PReLU()(x)
x = Flatten()(x)
x = Dense(2)(x)
ip1 = PReLU(name='ip1')(x)
ip2 = Dense(num_classes, activation='softmax')(ip1)


model = Model(inputs=inputs, outputs=[ip2])
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

if isCenterloss:
  input_target = Input(shape=(10,))
  centers = Embedding(10,2)(input_target)
  l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([ip1,centers])
  model_centerloss = Model(inputs=[inputs,input_target],outputs=[ip2,l2_loss])        
  model_centerloss.compile(optimizer=SGD(lr=0.05), loss=["categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,0.2],metrics=['accuracy'])


# prepare callback
histories = TYY_callbacks.Histories(isCenterloss)

# fit
if isCenterloss:
  random_y_train = np.random.rand(x_train.shape[0],1)
  random_y_test = np.random.rand(x_test.shape[0],1)
  
  model_centerloss.fit([x_train,y_train], [y_train, random_y_train], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([x_test,y_test], [y_test,random_y_test]), callbacks=[histories])

else:
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks=[histories])

