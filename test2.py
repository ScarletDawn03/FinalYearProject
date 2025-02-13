import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    
    # LSTM Layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Fully connected output layer
    model.add(Dense(units=1))  # Predicting one value (the next day's closing price)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define input shape based on X_tPPrain
input_shape = (X_train.shape[1], X_train.shape[2])

# Create model
model = create_lstm_model(input_shape)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions on the test data
predictions = model.predict(X_test)
