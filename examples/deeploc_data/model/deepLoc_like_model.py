from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, InputLayer,concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils import np_utils
from keras import optimizers
from keras_self_attention import SeqSelfAttention
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
import keras.layers

max_features = 26
embedding_size = 256
filters = 20
kernel_size = 5
hidden_dims = 650
lstm_output_size = 64
pool_size = 2
input_length = 2000

model1 = Sequential()
model1.add(Embedding(max_features, embedding_size, input_length=input_length))

model2_out = Conv1D(filters,3,padding = 'same',activation = 'relu',strides = 1)(model1.output)

model3_out = Conv1D(filters,5,padding = 'same',activation = 'relu',strides = 1)(model1.output)

concatenated = concatenate([model2_out, model3_out])

newOut = Conv1D(filters*2,3,padding = 'same',activation = 'relu',strides = 1)(concatenated)

newOut = Bidirectional(LSTM(units=lstm_output_size, return_sequences=True))(newOut)
newOut = SeqSelfAttention(attention_activation='sigmoid')(newOut)
newOut = Flatten()(newOut)
newOut = Dense(10)(newOut)

model = Model(inputs=model1.input,outputs=newOut)



#
#
#
#print('Building model...')
#model = Sequential()
#model.add(Embedding(max_features, embedding_size, input_length=input_length))
##model.add(Dropout(0.25))
#model.add(Conv1D(filters,3,padding = 'same',activation = 'relu',strides = 1))
#model.add(GlobalMaxPooling1D())
#model.add(Dense(hidden_dims))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
#
#model.add(MaxPooling1D(pool_size = pool_size))
#
#model.add(Bidirectional(LSTM(units=lstm_output_size, return_sequences=True)))
#model.add(SeqSelfAttention(attention_activation='sigmoid'))
#
#model.add(Flatten())
#
#model.add(Dense(10))
#model.add(Activation('softmax'))
#
#model.compile(loss = 'categorical_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
model.summary()

#model = Sequential()
#model.add(Embedding(max_features, embedding_size, input_length = 2000))
#model.add(Dropout(0.25))
#model.add(Conv1D(filters, kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
#model.add(GlobalMaxPooling1D())
#model.add(Dense(hidden_dims,name="dense"))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
#model.add(Dense(6))
#model.add(Activation('softmax'))
#model.compile(loss = 'categorical_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
#model.summary()