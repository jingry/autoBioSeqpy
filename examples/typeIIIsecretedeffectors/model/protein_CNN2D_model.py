'''
Trains a simple convnet for protein classification.
(there is still a lot of margin for parameter tuning)

'''

#from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers


print('Building model...')
model = Sequential()
model.add(Conv2D(250, kernel_size = (20,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (1, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(650, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

