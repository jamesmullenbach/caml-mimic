"""
    Use these methods to build CNN models
"""
import constants
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential

from constants import *

def vanilla(Y, filter_size, conv_dim_factor, loss_func="binary_crossentropy", padding="var"):
    """
        Builds the single window CNN model
        params:
            Y: size of the label space
        returns:
            cnn: the CNN model
    """
    cnn = Sequential()
    if padding != "var":
        cnn.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, input_length=MAX_LENGTH, init=EMBED_INIT))
    else:
        cnn.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, init=EMBED_INIT))
    cnn.add(Convolution1D(Y*conv_dim_factor, filter_size, activation=ACTIVATION_CONV, border_mode=WINDOW_TYPE))
    cnn.add(GlobalMaxPooling1D())
    cnn.add(Dense(Y, activation="tanh"))
    cnn.add(Dropout(DROPOUT_DENSE))
    cnn.add(Activation('sigmoid'))
    cnn.compile(optimizer=OPTIMIZER, loss=loss_func)
    print(cnn.summary())
    print
    return cnn 

def multi_window(Y, min_filter, max_filter, conv_dim_factor, padding="var"):
    """
        Builds the multi-window CNN model
        params:
            Y: size of the label space
            s: the smallest filter size
            l: the largest filter size
            step: size difference between consecutive filters
        returns:
            cnn_multi: the model
    """
    from keras.layers import Merge

    convs = []

    #set up shared embedding layer
    if padding != "var":
        base_embed = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, input_length=MAX_LENGTH, init=EMBED_INIT)
    else:
        base_embed = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, init=EMBED_INIT)
    for i,sz in enumerate(range(min_filter, max_filter+1)):
        convs.append(Sequential())
        convs[i].add(base_embed)

    #add the conv layers
    for i,sz in enumerate(range(min_filter, max_filter+1)):
        convs[i].add(Convolution1D(Y*conv_dim_factor, sz, activation='tanh'))
        from keras.layers.pooling import GlobalMaxPooling1D
        convs[i].add(GlobalMaxPooling1D())

    merged = Merge(convs, mode='concat', concat_axis=1) 

    cnn_multi = Sequential()
    cnn_multi.add(merged)
    cnn_multi.add(Dense(Y))
    cnn_multi.add(Dropout(DROPOUT_DENSE))
    cnn_multi.add(Activation('sigmoid'))
    
    cnn_multi.compile(optimizer=OPTIMIZER, loss=LOSS)
    print(cnn_multi.summary())
    print
    return cnn_multi

def lstm(Y):
    """
        Builds the LSTM model
        params:
            Y: size of the label space
        returns:
            lstm: the model
    """
    lstm = Sequential()
    lstm.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, init=EMBED_INIT))
    lstm.add(LSTM(EMBEDDING_SIZE, dropout_W=DROPOUT_DENSE, dropout_U=DROPOUT_DENSE))
    lstm.add(Dense(Y, activation="tanh"))
    lstm.add(Dropout(DROPOUT_DENSE))
    lstm.add(Activation("sigmoid"))

    lstm.compile(optimizer=OPTIMIZER, loss=LOSS)
    print(lstm.summary())
    print
    return lstm

