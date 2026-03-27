import streamlit as st
import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess_data
from src.feature_engineering import add_time_series_features
from src.train_model import train_model
from src.predict import predict_failure

DATA_PATH = "data/sample_sensor_data.csv"


@st.cache_data
def load_and_prepare_data():
    df = load_data(DATA_PATH)
    df_processed, scaler = preprocess_data(df)
    df_features = add_time_series_features(df_processed)
    return df_processed, df_features


@st.cache_resource
def get_model(df_features):
    return train_model(df_features)


# ✅ FIXED: model is now ignored in hashing
@st.cache_data
def get_predictions(_model, df_features):
    return predict_failure(_model, df_features)


def run_dashboard():

    st.title("Predictive Maintenance Dashboard")

    # Load + preprocess
    df_processed, df_features = load_and_prepare_data()

    # Train / load model
    model = get_model(df_features)

    # Get predictions
    predictions = get_predictions(model, df_features)

    # ---------------- VISUALS ---------------- #

    st.subheader("Sensor Trends")

    fig, ax = plt.subplots()

    ax.plot(df_processed["cycle"], df_processed["sensor1"], label="Sensor1")
    ax.plot(df_processed["cycle"], df_processed["sensor2"], label="Sensor2")
    ax.plot(df_processed["cycle"], df_processed["sensor3"], label="Sensor3")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Normalized Sensor Value")
    ax.legend()

    st.pyplot(fig)

    # ---------------- OUTPUTS ---------------- #

    st.subheader("Failure Probability")
    st.line_chart(predictions["failure_probability"])

    st.subheader("Equipment Health Score")
    st.line_chart(predictions["health_score"])

    st.subheader("Feature Importance (SHAP)")
    st.image("shap_feature_importance.png")
