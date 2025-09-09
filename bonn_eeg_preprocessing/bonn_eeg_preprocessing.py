import numpy as np
import pandas as pd
import os
import pywt

rows = []
labels = []
dirs = ['S','F','N','O','Z']
for d in dirs:
    for f in os.listdir(d):
        v = pd.read_csv(f'{d}/{f}', names=['v'])['v'].to_numpy()
        coeffs = pywt.wavedec(v, 'db4', level=4, mode='periodization')
        rows.append(np.concatenate(coeffs))
        labels.append(d)

df = pd.DataFrame(np.vstack(rows))
df['label'] = labels
df.to_csv("bonn_eeg_data.csv", index=False)