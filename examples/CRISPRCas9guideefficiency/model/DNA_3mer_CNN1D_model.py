
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers



max_features = 100
embedding_size = 256
filters = 100
kernel_size = 3 
pool_size = 2
hidden_dims1 = 80
hidden_dims2 = 40
hidden_dims3 = 20



print('Building model...')
model = Sequential()
# here you need to specify the input_length, we set the value of 28 as an example
model.add(Embedding(max_features,embedding_size,input_length=28))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Flatten())
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

