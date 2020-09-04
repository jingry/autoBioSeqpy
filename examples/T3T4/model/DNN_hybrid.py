from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers

model = Sequential()
model.add(Dense(210, activation = 'relu', input_shape=(420,)))
model.add(Dropout(0.2))
model.add(Dense(105, activation = 'relu'))
model.add(Dropout(0.2))


