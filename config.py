""" Definition of all constants used in main.ipynb """

RANDOM_SEED         = 16                # Seed for data splitting
NUM_FEATURES        = 20000             # Number of top most frequent features to be used in the ANN 
MAX_SEQUENCE_LENGTH = 250               # Limit on the length of text sequences (Longer sequences will be truncated)
TRAIN_PERCENTAGE    = 0.7               # Percentage of the dataset to retain for learning
VAL_PERCENTAGE      = 0.15              #           -||-          -||-           validation
EMBEDDING_FILE      = './crawl-300d-2M.vec/crawl-300d-2M.vec' # File with embeddings. Source: https://fasttext.cc/docs/en/english-vectors.html
EMBEDDING_DIMENSION = 300               # Dimension of word embeddings according to the file defined above
EMBEDDING_TRAINABLE = False             # Whether or not to train the embedding layer
EPOCHS              = 30                # Epochs to train the model
DROPOUT_RATE        = 0.3               # Rate for the dropour layers
BLOCKS              = 4                 # No. blocks for CNN
FILTERS             = [16, 32, 64, 128] # No filters for the CONV layers of the CNN
KERNEL_SIZE         = 3                 # Kernel size of the convolutions
POOL_SIZE           = 3                 # Max pooling size
LR_PLATEAU_FACTOR   = 0.1               # Multiplication factor for the learning rate scheduler (reduce on plateau)
LR_PLATEAU_PATIENCE = 5                 # No. epochs to wait for LR reduction
LR_PLATEAU_MIN      = 1e-6              # Min. learning rate
LRELU_ALPHA         = 0.2               # LeakyRelu alpha parameter
LEARN_RATE          = 1e-3              # Initial learning rate
BATCH_SIZE          = 512               # Training batch size
DENSE_UNITS         = 128               # No. units of the dense layer (prior to the head classifier)
CLASS_WEIGHTS       = {0: 1.0, 1: 12.5} # Class weights for imbalance
CHECKPOINT_DIR      = "./checkpoints/checkpoint_Epoch_{epoch}.keras" # Checkpoint directory