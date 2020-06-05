from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, AveragePooling1D
from keras import optimizers

filters = 80
kernel_size = 5 
pool_size = 2
hidden_dims1 = 80
hidden_dims2 = 40
hidden_dims3 = 40



print('Building model...')
model = Sequential()
model.add(Reshape((30,4),input_shape=(120,)))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(AveragePooling1D(pool_size = pool_size))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(hidden_dims1))
model.add(Dropout(0.3))
model.add(Dense(hidden_dims2))
model.add(Dropout(0.3))
model.add(Dense(hidden_dims3))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

