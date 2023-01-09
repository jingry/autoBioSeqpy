from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import GRU, Bidirectional, Flatten
from keras import optimizers


max_features = 26
embedding_size = 128
filters = 250
kernel_size = 15
lstm_output_size = 64
hidden_dims = 125

print('Building model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size,input_length=2000))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(Bidirectional(GRU(lstm_output_size,return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))


