import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters


def run_tsfresh(input_series_train, input_series_test):
    train_df = pd.DataFrame(input_series_train).copy()
    test_df = pd.DataFrame(input_series_test).copy()
    train_df['id'] = range(len(train_df))
    test_df['id'] = range(len(test_df))
    train_long = train_df.melt(id_vars=['id'], var_name='time', value_name='value')
    test_long = test_df.melt(id_vars=['id'], var_name='time', value_name='value')
    fc = MinimalFCParameters()
    train_features = extract_features(train_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=1, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    test_features = extract_features(test_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=1, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    return test_features, train_features


