from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, InputLayer,concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils import np_utils
from keras import optimizers
#from keras_self_attention import SeqSelfAttention
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
import keras.layers
import tensorflow as tf



max_features = 400
embedding_size = 256
filters = 50
kernel_size = 5
hidden_dims = 650
lstm_output_size = 64
pool_size = 2
input_length = 798

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Reshape((800,20), input_shape=(800*20,)))

#model1.add(tf.keras.layers.Embedding(max_features, embedding_size, input_length=input_length))
#model1.add(tf.keras.layers.Dropout(0.25))
model1_out = model1.output
model1_out = tf.keras.layers.Permute((2,1))(model1_out)

model2_out = tf.keras.layers.Conv1D(filters,3,padding = 'same',activation = 'relu',strides = 1)(model1_out)
#model2_out = tf.keras.layers.Activation('relu')(model2_out)
#model2_out = tf.keras.layers.MaxPooling1D(pool_size = pool_size)(model2_out)

model3_out = tf.keras.layers.Conv1D(filters,5,padding = 'same',activation = 'relu',strides = 1)(model1_out)
#model3_out = tf.keras.layers.Activation('relu')(model3_out)
#model3_out = tf.keras.layers.MaxPooling1D(pool_size = pool_size)(model3_out)

concatenated = tf.keras.layers.concatenate([model2_out, model3_out])

newOut = tf.keras.layers.Conv1D(filters*2,3,padding = 'same',activation = 'relu',strides = 1)(concatenated)
newOut = tf.keras.layers.MaxPooling1D(pool_size = pool_size)(newOut)

newOut = tf.keras.layers.Permute((2,1))(newOut)

#lstmFwOut = tf.keras.layers.LSTM(units=30,return_sequences=True,activation='tanh')(newOut)
##lstmFwOut = tf.keras.layers.Activation('sigmoid')(lstmFwOut)
#
#lstmBwOut = tf.keras.layers.LSTM(units=30,return_sequences=True,go_backwards=True,activation='tanh')(newOut)
##lstmBwOut = tf.keras.layers.Activation('sigmoid')(lstmBwOut)
#
#newOut = tf.keras.layers.Attention(causal=False)([lstmFwOut,lstmBwOut])


newOut = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=30,return_sequences=True,activation='tanh'))(newOut)
newOut = tf.keras.layers.Attention(causal=False)(newOut)

newOut = tf.keras.layers.Activation('sigmoid')(newOut)
#newOut = tf.keras.layers.Permute((2,1))(newOut)

#newOut = tf.keras.layers.Conv1D(filters,3,padding = 'same',activation = 'relu',strides = 1)(newOut)

#newOut = Bidirectional(LSTM(units=lstm_output_size, return_sequences=True))(newOut)
#newOut = SeqSelfAttention(attention_activation='sigmoid')(newOut)

#newOut = tf.keras.layers.Cropping1D(cropping=(1,filters*2-2))(newOut)
#newOut = tf.keras.layers.Flatten()(newOut)
#model.add(MaxPooling1D(pool_size = pool_size))
newOut = tf.keras.layers.GlobalMaxPooling1D()(newOut)

#newOut = tf.keras.layers.Dense(400)(newOut)
#newOut = tf.keras.layers.Dropout(0.5)(newOut)
#newOut = tf.keras.layers.Activation('relu')(newOut)

newOut = tf.keras.layers.Dense(30)(newOut)
#newOut = tf.keras.layers.Dropout(0.5)(newOut)
newOut = tf.keras.layers.Activation('relu')(newOut)

newOut = tf.keras.layers.Dense(10)(newOut)
newOut = tf.keras.layers.Activation('softmax')(newOut)

model = tf.keras.Model(inputs=model1.input,outputs=newOut)
model.summary()
#custom_objects=SeqSelfAttention.get_custom_objects()

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
#model.summary()

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