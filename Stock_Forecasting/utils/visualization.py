import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler

# Create a logger with a specific name
logger = logging.getLogger(__name__)

# Configure logger settings (you can customize these settings)
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(asctime)s [%(levelname)s] %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def plot_predictions(stock_symbol, model, X_test, y_open_test, y_close_test, y_open_train, y_close_train):
    try:
        # Visualize predictions
        open_price_pred, close_price_pred = model.predict(X_test)

        # Create a separate scaler for open and close prices
        scaler_open = MinMaxScaler()
        scaler_close = MinMaxScaler()

        # Fit the scalers on the training data
        scaler_open.fit(y_open_train.reshape(-1, 1))
        scaler_close.fit(y_close_train.reshape(-1, 1))

        # Inverse transform the scaled predictions and true values
        open_price_pred = scaler_open.inverse_transform(open_price_pred)
        close_price_pred = scaler_close.inverse_transform(close_price_pred)
        y_open_true = scaler_open.inverse_transform(y_open_test.reshape(-1, 1))
        y_close_true = scaler_close.inverse_transform(y_close_test.reshape(-1, 1))

        # Plot the predictions vs. true values for both open and close prices
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(y_open_true, label="True Open Prices", color='blue')
        plt.plot(open_price_pred, label="Predicted Open Prices", color='red')
        plt.title(f"{stock_symbol} Open Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y_close_true, label="True Close Prices", color='blue')
        plt.plot(close_price_pred, label="Predicted Close Prices", color='red')
        plt.title(f"{stock_symbol} Close Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error plotting predictions: {e}", exc_info=True)
