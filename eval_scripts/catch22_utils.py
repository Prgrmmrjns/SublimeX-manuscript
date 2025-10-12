import numpy as np
from pycatch22 import catch22_all


def run_catch22(input_series_train, input_series_test):
    train_features = []
    for i in range(len(input_series_train)):
        features = catch22_all(input_series_train[i])['values']
        train_features.append(features)
    train_features = np.array(train_features)
    
    test_features = []
    for i in range(len(input_series_test)):
        features = catch22_all(input_series_test[i])['values']
        test_features.append(features)
    test_features = np.array(test_features)
    
    return test_features, train_features


