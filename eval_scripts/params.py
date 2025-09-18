## patx params

MAX_N_TRIALS = 300  # High number, will be stopped early by no improvement
N_JOBS = -1 # Single thread to avoid segmentation faults  / multiple threads only works for mitbih dataset
SHOW_PROGRESS = False
TEST_SIZE = 0.3
VAL_SIZE = 0.2
POLYNOMIAL_DEGREE = 3  # Degree of polynomial patterns (0=constant, 1=linear, 2=quadratic, etc.)

## tsfresh params
TSFRESH_N_JOBS = 1  # Number of jobs for tsfresh feature extraction

## cnn params
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3
CNN_PATIENCE = 10  # Early stopping patience

## other params