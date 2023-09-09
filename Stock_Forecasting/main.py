import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from utils.data_processing import download_stock_data, preprocess_data, prepare_sequences, split_data
from utils.model import build_model, compile_and_train_model
from utils.evaluation import evaluate_model
from utils.visualization import plot_predictions

if __name__ == "__main__":
    stock_symbol = "AAPL"
    start_date = "2000-01-01"
    end_date = "2023-09-08"
    n_steps = 60
    split_ratio = 0.8

    # Download and preprocess data
    stock_data = download_stock_data(stock_symbol, start_date, end_date)
    features_scaled = preprocess_data(stock_data)
    X, y_open, y_close = prepare_sequences(features_scaled, n_steps)
    X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test = split_data(X, y_open, y_close, split_ratio)

    # Build and train the model
    model = build_model(n_steps)
    compile_and_train_model(model, X_train, y_open_train, y_close_train, X_test, y_open_test, y_close_test)

    # Evaluate the model
    evaluate_model(model, X_test, y_open_test, y_close_test)

    # Visualize predictions
    plot_predictions(stock_symbol, model, y_open_test, y_close_test)
