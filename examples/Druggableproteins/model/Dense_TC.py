from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers

model = Sequential()
model.add(Dense(4000, activation = 'relu', input_shape=(8000,)))
model.add(Dropout(0.2))
model.add(Dense(2000, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

