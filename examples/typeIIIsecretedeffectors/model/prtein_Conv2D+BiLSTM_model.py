'''
Trains a simple convnet for protein classification.
(there is still a lot of margin for parameter tuning)

'''

#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LSTM, Bidirectional
from keras.layers import Reshape
from keras import optimizers


lstm_output_size = 64

model = Sequential()
model.add(Conv2D(250, kernel_size = (20,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (1,2)))
# 250 is the number of filters, 48 is (spcLen-kernel_size+1)/2,here spcLen 
# is 100
model.add(Reshape((250,48)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

