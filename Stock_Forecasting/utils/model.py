from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_stock_prediction_model(n_steps, num_features, filters=64, kernel_size=3, lstm_units=50):
    """
    Build and return a stock price prediction model.

    Args:
        n_steps (int): Number of time steps in the input sequence.
        num_features (int): Number of input features.
        filters (int, optional): Number of filters in the Conv1D layer. Defaults to 64.
        kernel_size (int, optional): Kernel size in the Conv1D layer. Defaults to 3.
        lstm_units (int, optional): Number of LSTM units. Defaults to 50.

    Returns:
        tensorflow.keras.Model: A compiled Keras model for stock price prediction.
    """
    try:
        # Input layer
        input_layer = Input(shape=(n_steps, num_features))

        # Convolutional layer
        conv1d_layer = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_layer)
        maxpooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)

        # LSTM layer
        lstm_layer = LSTM(units=lstm_units, activation='relu')(maxpooling_layer)

        # Output layers
        open_price_output = Dense(1, name='open_price')(lstm_layer)
        close_price_output = Dense(1, name='close_price')(lstm_layer)

        # Create and compile the model
        model = Model(inputs=input_layer, outputs=[open_price_output, close_price_output])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    except Exception as e:
        print(f"Error building the model: {e}")
        return None
