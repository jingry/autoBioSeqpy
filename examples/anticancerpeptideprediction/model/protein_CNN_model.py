from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, AveragePooling1D
from keras import optimizers



max_features = 26
embedding_size = 256
filters = 250
kernel_size = 5 
pool_size = 2
hidden_dims = 650



print('Building model...')
model = Sequential()
model.add(Embedding(max_features,embedding_size,input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(AveragePooling1D(pool_size = pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

