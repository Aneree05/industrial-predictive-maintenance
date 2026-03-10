import numpy as np
import pandas as pd

def predict_failure(model, df):

    X = df.drop(columns=["failure"])

    probs = model.predict_proba(X)[:, 1]

    df["failure_probability"] = probs

    df["health_score"] = (1 - probs) * 100

    return df[["cycle", "failure_probability", "health_score"]]