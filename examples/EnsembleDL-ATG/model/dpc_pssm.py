from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers


print('Building model...')
model = Sequential()
model.add(Dense(350, activation = 'relu', input_shape=(400,)))
model.add(Dropout(0.2))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation = 'relu'))

