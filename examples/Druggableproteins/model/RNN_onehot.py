from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Reshape
from keras.layers import LSTM, Bidirectional
from keras import optimizers




lstm_output_size = 16


print('Building model...')
model = Sequential()
model.add(Reshape((500,20),input_shape=(10000,)))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
