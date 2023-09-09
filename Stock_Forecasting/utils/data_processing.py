import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Create a logger with a specific name
logger = logging.getLogger(__name__)

# Configure logger settings (you can customize these settings)
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(asctime)s [%(levelname)s] %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def download_stock_data(symbol, start_date, end_date):
    try:
        # Download and return historical stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error downloading stock data: {e}", exc_info=True)
        return None

def preprocess_data(data):
    try:
        # Preprocess data and return scaled features
        features = data[['Open', 'Close', 'High', 'Low', 'Volume']].values
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        return features_scaled
    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error preprocessing data: {e}", exc_info=True)
        return None

def prepare_sequences(features_scaled, n_steps):
    try:
        # Prepare input sequences and target variables
        X, y_open, y_close = [], [], []
        for i in range(n_steps, len(features_scaled)):
            X.append(features_scaled[i - n_steps:i, :])
            y_open.append([features_scaled[i, 0]])  # Opening price
            y_close.append([features_scaled[i, 1]])  # Closing price
        return np.array(X), np.array(y_open), np.array(y_close)
    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error preparing sequences: {e}", exc_info=True)
        return None, None, None

def split_data(X, y_open, y_close, split_ratio):
    try:
        # Split data into training and testing sets
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_open_train, y_open_test = y_open[:split_index], y_open[split_index:]
        y_close_train, y_close_test = y_close[:split_index], y_close[split_index:]
        return X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test
    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error splitting data: {e}", exc_info=True)
        return None, None, None, None, None, None
