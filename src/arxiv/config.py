

###############################################
###############################################
#############   Training Config ###############
###############################################
###############################################
NUM_EPOCHS = 200
NUM_EPOCHS_LP = 100
NUM_EPOCHS_LOGIC = 200
NUM_EPOCHS_MLP = 200
HIDDEN_DIMS = [[128, 128, 128]]
STARTING_LR = 1e-3
EMBED_DIM = 32
GRAPH_TASK = 0.5
LOGIC_TASK = 0.5
NUM_NEIGHBORS = [-1]
BATCH_SIZE= 128
EPS=1e-15

###############################################
###############################################
#############   Data           ################
###############################################
###############################################
LARGE_DIR = "large_graphs"
RAW_FILE_NAME = "pair_data.txt"
FILE_DIR = "graphdata"
FILE_LOGIC_DIR = "graphdata_logic"
SEED_MEANS = [10, 30, 15, 5]
N_ELEMENTS = 5000
N_SETS=50 # sets for set dataset class
N_STATEMENTS=20
DATA_DIM = 2 # original set data dimension
STD_DIR = 3 # standard deviation
STD_DIR_MIN = 0.1 # minimum multiplier
STD_DIR_MULTIPLIER = 0.3 # max multiplier
SIZE_MULTIPLIER = 0.05 # max set size as a fraction of total samples 
POWER_LAW_COEFF = 7
MAX_X1 = 36.0
MIN_X1 = 0.0
MAX_X2 = 36.0
MIN_X2 = 0.0
N_PAIR_ELEMS = 50
MARGIN = 1.0
MIN_RAD_MULTIPLIER = 5 # min radius related