# optuna params
N_TRIALS = 10000  
N_STARTUP_TRIALS = 1000
EARLY_STOPPING_PATIENCE = 1000
SHOW_PROGRESS = True
N_WORKERS = -1

# pattern params
N_PATTERNS = 15
N_CONTROL_POINTS = 5
N_TRANSFORMS = 5
MAX_SAMPLES = 3000

# Validation params
K_FOLDS = 5
INNER_K_FOLDS = 3
TEST_SIZE = 0.3 # Test size for datasets that don't use K-fold cross-validation
VAL_SIZE = 0.2

# CNN params
CNN_EPOCHS = 100
CNN_LEARNING_RATE = 0.1



