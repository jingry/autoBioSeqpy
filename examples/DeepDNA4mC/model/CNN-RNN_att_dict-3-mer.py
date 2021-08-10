from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras import optimizers
from keras_self_attention import SeqSelfAttention
from keras.layers import Flatten

max_features = 100
embedding_size = 64
filters = 150
kernel_size = 3
pool_size = 2
lstm_output_size = 128

print('Building model...')
model = Sequential()
model.add(Embedding(max_features,embedding_size,input_length=39))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

custom_objects = SeqSelfAttention.get_custom_objects()

