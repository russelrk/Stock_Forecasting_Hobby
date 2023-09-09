import matplotlib.pyplot as plt

def plot_predictions(stock_symbol, model, y_open_test, y_close_test):
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
    plt.xlabel("Time
