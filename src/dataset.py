import numpy as np


def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
