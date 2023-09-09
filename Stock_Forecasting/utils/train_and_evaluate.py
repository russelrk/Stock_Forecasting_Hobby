from tensorflow.keras.optimizers import Adam

def compile_and_train_model(model, X_train, y_open_train, y_close_train, X_test, y_open_test, y_close_test):
    # Compile and train the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, [y_open_train, y_close_train], epochs=50, batch_size=32, validation_data=(X_test, [y_open_test, y_close_test]), verbose=1)


def evaluate_model(model, X_test, y_open_test, y_close_test):
    # Evaluate and print model performance
    loss = model.evaluate(X_test, [y_open_test, y_close_test], verbose=0)
    print(f"Total Loss: {loss[0]}")
    print(f"Open Price Loss: {loss[1]}")
    print(f"Close Price Loss: {loss[2]}")
