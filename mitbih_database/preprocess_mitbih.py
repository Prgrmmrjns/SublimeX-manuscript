import os
import csv
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

num_bins = 100

def preprocess_mitbih(num_bins):
    path = '.'

    classes = ['N', 'L', 'R', 'A', 'V']
    # N: Normal beat
    # L: Left bundle branch block beat
    # R: Right bundle branch block beat
    # A: Atrial premature beat
    # V: Premature ventricular contraction
    n_classes = len(classes)
    count_classes = [0]*n_classes

    X = list()
    y = list()

    filenames = next(os.walk(path))[2]

    records = list()
    annotations = list()
    filenames.sort()

    # Segregating filenames and annotations
    for f in filenames:
        filename, file_extension = os.path.splitext(f)
        
        # *.csv
        if(file_extension == '.csv'):
            records.append(filename + file_extension)

        # *.txt
        else:
            annotations.append(filename + file_extension)

    # Records
    for r in range(0, len(records)):
        signals = []
        with open(records[r], 'rt') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') 
            row_index = -1
            for row in spamreader:
                if(row_index >= 0):
                    signals.insert(row_index, int(row[1]))
                row_index += 1
        with open(annotations[r], 'r') as fileID:
            data = fileID.readlines() 
            beat = list()
            for d in range(1, len(data)): 
                splitted = data[d].split(' ')
                splitted = list(filter(None, splitted))
                if len(splitted) < 3:
                    continue
                # Skip the first element if it's not needed
                splitted = splitted[1:]
                pos = int(splitted[0]) 
                arrhythmia_type = splitted[1]
                if(arrhythmia_type in classes):
                    arrhythmia_index = classes.index(arrhythmia_type)
                    count_classes[arrhythmia_index] += 1
                    if(num_bins <= pos < (len(signals) - num_bins)):
                        half_bins_before = num_bins // 2
                        half_bins_after = num_bins - half_bins_before
                        # Take half_bins_before before and half_bins_after after
                        beat = signals[pos - half_bins_before: pos] + signals[pos: pos + half_bins_after]    
                        X.append(beat)
                        y.append(arrhythmia_index)

    for i in range(0, len(X)):
            X[i] = np.append(X[i], y[i])

    data = pd.DataFrame(X)
    return data

data = preprocess_mitbih(num_bins=num_bins)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)
X_res_df = pd.DataFrame(X_res, columns=X.columns)
y_res_series = pd.Series(y_res, name='target')
data = pd.concat([X_res_df, y_res_series], axis=1)

data.to_csv(f"mitbih_processed.csv", index=False)