from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers


print('Building model...')
model = Sequential()
model.add(Dense(140, activation = 'relu', input_shape=(210,)))
model.add(Dropout(0.2))
model.add(Dense(70, activation = 'relu'))

