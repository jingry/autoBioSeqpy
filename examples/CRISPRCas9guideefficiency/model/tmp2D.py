from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras.layers import Conv2D, AveragePooling2D
from keras import optimizers

print('Building model...')
model = Sequential()
model.add(Reshape((4,30,1), input_shape=(120,)))
model.add(Conv2D(80, kernel_size = (4,5),activation = 'relu'))
model.add(AveragePooling2D(pool_size = (1,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

#model1 = Sequential()
#model1.add(Reshape((8,15,1), input_shape=(120,)))
#model1.add(model)
