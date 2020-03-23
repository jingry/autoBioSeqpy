from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras import optimizers




max_features = 26
embedding_size = 256



#print('Building model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(64,dropout=0.2,recurrent_dropout = 0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

