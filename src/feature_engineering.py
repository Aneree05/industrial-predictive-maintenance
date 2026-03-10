import numpy as np
import pandas as pd

def add_time_series_features(df):
    df = df.copy()

    sensors = ["sensor1", "sensor2", "sensor3"]

    for col in sensors:
        df[f"{col}_roll_mean"] = df[col].rolling(window=3).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=3).std()
        df[f"{col}_min"] = df[col].rolling(window=3).min()
        df[f"{col}_max"] = df[col].rolling(window=3).max()

        fft_vals = np.abs(np.fft.fft(df[col]))
        df[f"{col}_fft"] = fft_vals

    df.fillna(0, inplace=True)

    return df