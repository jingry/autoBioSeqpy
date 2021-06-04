from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers


filters = 50
kernel_size = 3
pool_size = 10
hidden_dims = 650



print('Building model...')
model = Sequential()
model.add(Reshape((2000,20),input_shape=(40000,)))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

