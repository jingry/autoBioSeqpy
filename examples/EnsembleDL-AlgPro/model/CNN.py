from keras.models import Sequential
from keras.layers import Reshape, Dropout,Conv1D,GlobalMaxPooling1D,Dense


print('Building model...')
model = Sequential()
model.add(Reshape((1000,20),input_shape=(20000,)))
model.add(Conv1D(250, 11, padding="valid", activation="relu", strides=1))
model.add(Conv1D(250, 11, padding="valid", activation="relu", strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128,activation="relu"))

