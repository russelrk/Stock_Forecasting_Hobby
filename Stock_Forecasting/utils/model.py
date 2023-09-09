from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(n_steps):
    # Build and return the model
    input_layer = Input(shape=(n_steps, 5))
    conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    maxpooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)
    lstm_layer = LSTM(50, activation='relu')(maxpooling_layer)
    open_price_output = Dense(1, name='open_price')(lstm_layer)
    close_price_output = Dense(1, name='close_price')(lstm_layer)
    model = Model(inputs=input_layer, outputs=[open_price_output, close_price_output])
    return model


