from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import optimizers


hidden_dims1 = 10000
hidden_dims2 = 1000
hidden_dims3 = 100


print('Building model...')
model = Sequential()
model.add(Dense(hidden_dims1,activation = 'relu', input_shape=(40000,)))
model.add(Dropout(0.3))
model.add(Dense(hidden_dims2))
model.add(Dropout(0.3))
model.add(Dense(hidden_dims3))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
