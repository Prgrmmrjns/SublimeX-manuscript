import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters


def run_tsfresh(X_train, y_train, X_test, task_type='classification', val_size=0.5, random_state=42, n_jobs=1):
    Xtr_df = pd.DataFrame(X_train).copy()
    Xte_df = pd.DataFrame(X_test).copy()
    Xtr_df['id'] = range(len(Xtr_df)); Xte_df['id'] = range(len(Xte_df))
    tr_long = Xtr_df.melt(id_vars=['id'], var_name='time', value_name='value')
    te_long = Xte_df.melt(id_vars=['id'], var_name='time', value_name='value')
    t0 = time.time()
    fc = MinimalFCParameters()
    tr_f = extract_features(tr_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=n_jobs, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    te_f = extract_features(te_long, column_id='id', column_sort='time', column_value='value', impute_function=impute, n_jobs=n_jobs, default_fc_parameters=fc, show_warnings=False, disable_progressbar=True)
    X_tr, X_val, y_tr, y_val = train_test_split(tr_f, y_train, test_size=val_size, random_state=random_state, stratify=y_train if task_type=='classification' else None)
    return te_f, X_tr, X_val, y_tr, y_val, time.time() - t0


