#Embedding constants
VOCAB_SIZE = 40000
EMBEDDING_SIZE = 100
DROPOUT_EMBED = 0.5
EMBED_INIT = 'glorot_uniform'

#Convolution constants
#now cmdline inputs
#FILTER_SIZE = 5
#MULTI_WINDOW = True
#MIN_FILTER = 3
#MAX_FILTER = 5
#CONV_DIM_FACTOR = 10
ACTIVATION_CONV = 'tanh'
WINDOW_TYPE = 'valid'

#training constants
BATCH_SIZE = 64
#now cmdline inputs
#NUM_EPOCHS = 5

#other NN constants
DROPOUT_DENSE = 0.5
OPTIMIZER = 'rmsprop'
LOSS = 'binary_crossentropy'
LEARNING_RATE = 0.01
MOMENTUM = 0
MAX_LENGTH = 400

DATA_DIR = "/nethome/jmullenbach3/mimicdata"
