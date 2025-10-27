# patx params
N_TRIALS = 500
SHOW_PROGRESS = False
VERBOSE = False
N_CONTROL_POINTS = 4

# K-fold cross-validation
K_FOLDS = 5

# Test size for datasets that don't use K-fold cross-validation
TEST_SIZE = 0.3
VAL_SIZE = 0.2

## cnn params
CNN_EPOCHS = 100
CNN_LEARNING_RATE = 0.01

# Pattern shift tolerance (fraction of series length, 0-1)
MAX_SHIFT_EVALUATIONS = 10  # Max number of shift positions to evaluate