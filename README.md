# Predictive Maintenance for Industrial Equipment

Machine failures in industrial environments such as drilling rigs, compressors, pumps, and turbines lead to costly downtime. Predictive maintenance uses machine learning models to analyze sensor data and estimate the probability of equipment failure before it occurs.

This project demonstrates a predictive maintenance system that analyzes time-series sensor readings and predicts failure risk.

---

## Project Structure

predictive-maintenance

data/  
sample_sensor_data.csv

src/  
preprocess.py  
feature_engineering.py  
train_model.py  
predict.py  
dashboard.py  

main.py  
requirements.txt  
README.md  

---

## Features

Preprocessing

- Missing value handling
- Sensor normalization using StandardScaler

Feature Engineering

- Rolling mean
- Rolling standard deviation
- Rolling min and max
- FFT-based signal feature

Machine Learning Model

- XGBoost classifier
- Train/test split
- Accuracy evaluation

Prediction Output

- Failure probability
- Equipment health score (0–100)

Dashboard

Streamlit interface showing:

- Sensor trend charts
- Predicted failure probability
- Equipment health score
- SHAP feature importance visualization

---

## Installation

Clone the repository and install dependencies.

```bash
pip install -r requirements.txt