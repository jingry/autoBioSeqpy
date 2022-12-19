from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras import optimizers

model = Sequential()
model.add(Dense(708, activation = 'relu', input_shape=(1364,)))
model.add(Dropout(0.2))
model.add(Dense(354, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(177, activation='relu'))
