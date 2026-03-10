import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data
from feature_engineering import add_time_series_features
from train_model import train_model
from predict import predict_failure

DATA_PATH = "data/sample_sensor_data.csv"

st.title("Predictive Maintenance Dashboard")

# Load data
df = load_data(DATA_PATH)

# Preprocess
df_processed, scaler = preprocess_data(df)

# Feature engineering
df_features = add_time_series_features(df_processed)

# Train model
model = train_model(df_features)

# Predictions
predictions = predict_failure(model, df_features)

st.subheader("Sensor Trends")

fig, ax = plt.subplots()

ax.plot(df_processed["cycle"], df_processed["sensor1"], label="Sensor1")
ax.plot(df_processed["cycle"], df_processed["sensor2"], label="Sensor2")
ax.plot(df_processed["cycle"], df_processed["sensor3"], label="Sensor3")

ax.set_xlabel("Cycle")
ax.set_ylabel("Normalized Sensor Value")
ax.legend()

st.pyplot(fig)

st.subheader("Failure Probability")
st.line_chart(predictions["failure_probability"])

st.subheader("Equipment Health Score")
st.line_chart(predictions["health_score"])

st.subheader("Feature Importance (SHAP)")
st.image("shap_feature_importance.png")