from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import LSTM, Bidirectional
from keras import optimizers
from keras_self_attention import SeqSelfAttention
from keras.layers import Flatten

lstm_output_size = 128


print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
print(model.outputs)
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True,activation = 'tanh')))
print(model.outputs)
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

custom_objects=SeqSelfAttention.get_custom_objects()
