import os
import csv
import numpy as np
import sys

# Add the parent directory (where ReusableFunctions is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import combinations, product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import custom class for technical indicators
from ReusableFunctions.DataPreprocessing import DataPreprocessing

# LSTM hyperparameters
lstm_params_grid = {
    'neurons': [32, 64, 128],
    'dropout_rate': [0.5, 0.7],
    'learning_rate': [0.0005, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150],
    'activation': ['tanh', 'sigmoid', 'relu'],
    'optimizer': ['adam'],  # You can expand this if needed
    'sequence_length': [15],  # Fixed as per your requirement
}

def normalize_data(df):
    """
    Normalize selected combinations of technical indicators using MinMaxScaler.
    Returns the original dataframe and a dictionary of scaled datasets.
    """
    all_indicators = [
        '20MA', '50MA', '200MA', 'RSI', 'MACD', 'Signal_Line',
        'Upper_BB', 'Lower_BB', 'CCI', 'ATR', 'ROC', 'Williams_%R', 'OBV'
    ]

    # Generate all possible combinations of 5 indicators + 'Close'
    indicator_combinations = list(combinations(all_indicators, 5))
    all_scaled_data = {}  # Dictionary to store scaled data for each combination

    for selected_indicators in indicator_combinations:
        selected_features = ['Close'] + list(selected_indicators)  # Always include Close

        # Normalize selected features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[selected_features])

        # Store the scaled data
        all_scaled_data[selected_indicators] = scaled_data

    return df, all_scaled_data

def create_time_series_data(df, scaled_data, window_size=50):
    """
    Convert scaled time-series data into input/output sequences for the model.
    """
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

def split_data(X, y, train_size=0.7, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    """
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

from tensorflow.keras.layers import LSTM

def create_slstm_model(input_shape, params):
    model = Sequential()
    
    model.add(LSTM(
    units=params['neurons'],
    activation=params['activation'],  # keep your activation
    input_shape=(X_train.shape[1], X_train.shape[2]),
    dropout=params['dropout_rate'],
    recurrent_dropout=0.0,  # can also be adjusted
    return_sequences=False,
    use_bias=True,
    recurrent_activation="sigmoid"  # Required for compatibility with non-CuDNN
    ))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1))

    # Optimizer
    if params['optimizer'] == 'adam':
        opt = Adam(learning_rate=params['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer. Only 'adam' is implemented.")

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def calculate_accuracy(y_true, y_pred, threshold_percent=5):
    """
    Calculate custom accuracy metric: percentage of predictions within a given threshold of the actual values.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
    within_threshold = percentage_diff <= threshold_percent

    correct_count = np.sum(within_threshold)  # Count of correct predictions
    total_count = len(within_threshold)       # Total predictions
    accuracy_percentage = (correct_count / total_count) * 100

    print(f"Correct predictions within {threshold_percent}%: {correct_count} out of {total_count}")
    return accuracy_percentage

# Define tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'XOM', '1155.KL', '5681.KL', '6963.KL']

# Create output directory if it doesn't exist
output_folder = "stock_results"
os.makedirs(output_folder, exist_ok=True)

# Create hyperparameter combinations from the grid
keys, values = zip(*lstm_params_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# Loop through each ticker
for ticker in tickers:
    print(f"Running model for {ticker}...\n")
    data_preprocessing = DataPreprocessing(ticker=ticker)
    df_with_indicators = data_preprocessing.add_technical_indicators()

    df, all_scaled_data = normalize_data(df_with_indicators)
    output_file = os.path.join(output_folder, f"{ticker}_results_LSTM.csv")
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Ticker", "Indicators", "Neurons", "Dropout Rate", "Learning Rate", "Batch Size", "Epochs", 
                  "Optimizer", "Activation", "Sequence Length", "RMSE", "MAPE", "R2", "Accuracy", 
                  "Train Loss", "Validation Loss"]
        writer.writerow(header)

        for selected_indicators, scaled_data in all_scaled_data.items():
            for params in param_combinations:
                print(f"Training with indicators: {selected_indicators}, params: {params}")
                seq_len = params['sequence_length']
                X, y = create_time_series_data(df, scaled_data, window_size=seq_len)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                # Normalize target values
                scaler_y = MinMaxScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
                y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

                input_shape = (X_train.shape[1], X_train.shape[2])
                model = create_slstm_model(input_shape=input_shape, params=params)

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = model.fit(
                    X_train, y_train_scaled,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(X_val, y_val_scaled),
                    callbacks=[early_stopping],
                    verbose=0
                )

                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]

                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                y_test_original = scaler_y.inverse_transform(y_test_scaled)

                accuracy = calculate_accuracy(y_test_original, y_pred)
                mse = mean_squared_error(y_test_original, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100
                r2 = r2_score(y_test_original, y_pred)

                writer.writerow([
                    ticker, ', '.join(selected_indicators), params['neurons'], params['dropout_rate'], 
                    params['learning_rate'], params['batch_size'], params['epochs'], params['optimizer'], 
                    params['activation'], params['sequence_length'],
                    rmse, mape, r2, accuracy, final_train_loss, final_val_loss
                ])
                file.flush()

            print(f"{ticker} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%\n")
