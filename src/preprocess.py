import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.copy()

    # handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    sensor_cols = ["sensor1", "sensor2", "sensor3"]

    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    return df, scaler