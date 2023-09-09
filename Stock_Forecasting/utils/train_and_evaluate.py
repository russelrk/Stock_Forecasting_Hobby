from tensorflow.keras.optimizers import Adam
import logging

# Create a logger with a specific name
logger = logging.getLogger(__name__)

# Configure logger settings (you can customize these settings)
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(asctime)s [%(levelname)s] %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def compile_and_train_model(model, X_train, y_open_train, y_close_train, X_test, y_open_test, y_close_test, epochs=50, batch_size=32):
    """
    Compile and train the model.

    Args:
        model (tensorflow.keras.Model): The Keras model to compile and train.
        X_train (numpy.ndarray): Training input data.
        y_open_train (numpy.ndarray): Training target data for open prices.
        y_close_train (numpy.ndarray): Training target data for close prices.
        X_test (numpy.ndarray): Testing input data.
        y_open_test (numpy.ndarray): Testing target data for open prices.
        y_close_test (numpy.ndarray): Testing target data for close prices.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size for training. Defaults to 32.
    """
    try:
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, [y_open_train, y_close_train], epochs=epochs, batch_size=batch_size, validation_data=(X_test, [y_open_test, y_close_test]), verbose=1)

    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error during model training: {e}", exc_info=True)


def evaluate_model(model, X_test, y_open_test, y_close_test):
    """
    Evaluate and print model performance.

    Args:
        model (tensorflow.keras.Model): The Keras model to evaluate.
        X_test (numpy.ndarray): Testing input data.
        y_open_test (numpy.ndarray): Testing target data for open prices.
        y_close_test (numpy.ndarray): Testing target data for close prices.
    """
    try:
        # Evaluate the model
        loss = model.evaluate(X_test, [y_open_test, y_close_test], verbose=0)
        print(f"Total Loss: {loss[0]}")
        print(f"Open Price Loss: {loss[1]}")
        print(f"Close Price Loss: {loss[2]}")

    except Exception as e:
        # Log the exception with a stack trace
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
