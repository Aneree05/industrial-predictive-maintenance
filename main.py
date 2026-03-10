from src.preprocess import load_data, preprocess_data
from src.feature_engineering import add_time_series_features
from src.train_model import train_model
from src.predict import predict_failure
from src.dashboard import run_dashboard

DATA_PATH = "data/sample_sensor_data.csv"

def main():

    df = load_data(DATA_PATH)

    df_processed, scaler = preprocess_data(df)

    df_features = add_time_series_features(df_processed)

    model = train_model(df_features)

    predictions = predict_failure(model, df_features)

    run_dashboard(df_processed, predictions)


if __name__ == "__main__":
    main()