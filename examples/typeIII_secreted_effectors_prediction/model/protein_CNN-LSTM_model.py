from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers


max_features = 26
embedding_size = 256
kernel_size = 5
filters = 250
pool_size = 2
lstm_output_size = 64



#print('Building model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])


