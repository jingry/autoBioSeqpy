from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers


print('Building model...')
model = Sequential()
model.add(Dense(168, activation = 'relu', input_shape=(210,)))
model.add(Dropout(0.2))
model.add(Dense(126, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(84, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(42, activation = 'relu'))

