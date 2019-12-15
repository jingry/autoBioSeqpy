'''
# Trains and evaluate a simple MLP on classification task
'''
#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers


#print('Building model...')
model = Sequential()
model.add(Dense(1000, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(125, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])







