from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers
from keras_self_attention import SeqSelfAttention
from keras.layers import Flatten

max_features = 26
embedding_size = 256
kernel_size = 9
filters = 50
pool_size = 8
lstm_output_size = 16
input_length = 2000




model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=input_length))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Bidirectional(LSTM(units=lstm_output_size, return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
model.summary()

custom_objects = SeqSelfAttention.get_custom_objects()
