#Embedding constants
VOCAB_SIZE = "full"
EMBEDDING_SIZE = 100
CODE_EMBEDDING_SIZE = 32
DROPOUT_EMBED = 0.5
EMBED_INIT = 'glorot_uniform'
PAD_CHAR = "**PAD**"

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
BATCH_LENGTH = 256
MAX_NOTES = 100

#other NN constants
DROPOUT_DENSE = 0.5
OPTIMIZER = 'rmsprop'
LOSS = 'binary_crossentropy'
LEARNING_RATE = 0.01
MOMENTUM = 0
MLP_OUTPUT = 16

#DATA_DIR = "/nethome/jmullenbach3/mimicdata"
DATA_DIR = "/media/james/Windows8_OS/Users/James/Documents/SCHOOL/MS/research/mimic/mimicdata/with_hadm_id"
#MODEL_DIR = "/nethome/jmullenbach3/cnn-medical-text/saved_models"
MODEL_DIR = "/media/james/Windows8_OS/Users/James/Documents/SCHOOL/MS/research/mimic/cnn-medical-text/models"
