import os
import csv
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

from itertools import product

# Import the TechnicalIndicators class from the technical_indicators module
from ReusableFunctions.TechnicalIndicators import TechnicalIndicators

# Example usage of the TechnicalIndicators class
tech_indicators = TechnicalIndicators(ticker="AAPL")
df_with_indicators = tech_indicators.add_technical_indicators()

# Define ranges of hyperparameters to search
param_grid = {
    'units': [32, 64, 128, 256],
    'dropout_rate': [0.2, 0.4, 0.5, 0.6],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 200, 300],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'sequence_length': [10, 20, 30],
    'optimizer': ['adam']  # Only Adam for now
}

# Generate all combinations of hyperparameters
all_param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

# Function to normalize the data
def normalize_data(df):
    all_indicators = [
        '20MA', '50MA', '200MA', 'RSI', 'MACD', 'Signal_Line',
        'Upper_BB', 'Lower_BB', 'CCI', 'ATR', 'ROC', 'Williams_%R', 'OBV'
    ]

    # Generate all possible combinations of 5 indicators + 'Close'
    indicator_combinations = list(combinations(all_indicators, 5))

    # Dictionary to store performance results
    results = {}

    all_scaled_data = {}  # Dictionary to store scaled data for each combination

    for selected_indicators in indicator_combinations:
        selected_features = ['Close'] + list(selected_indicators)  # Always include Close

        # Normalize selected features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[selected_features])

        # Store the scaled data
        all_scaled_data[selected_indicators] = scaled_data

    return df, all_scaled_data

# Prepare the time-series data
def create_time_series_data(df, scaled_data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

# Split data into training and testing sets (80% training, 20% testing)
def split_data(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_lstm_model(input_shape, params):
    model = Sequential()
    model.add(LSTM(units=params['units'],
                   activation=params['activation'],
                   input_shape=input_shape,
                   return_sequences=False))
    model.add(Dropout(rate=params['dropout_rate']))
    model.add(Dense(units=1))

    # Optimizer setup
    if params['optimizer'] == 'adam':
        opt = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        opt = RMSprop(learning_rate=params['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer.")

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def calculate_accuracy(y_true, y_pred, threshold_percent=5):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
    within_threshold = percentage_diff <= threshold_percent

    correct_count = np.sum(within_threshold)  # Count of correct predictions within threshold
    total_count = len(within_threshold)  # Total number of predictions
    accuracy_percentage = (correct_count / total_count) * 100  # Calculate accuracy percentage

    # Optional print/logging for debugging or further analysis
    print(f"Correct predictions within {threshold_percent}%: {correct_count} out of {total_count}")
    return accuracy_percentage


# Define tickers
tickers = ['AAPL']

# Dictionary to store results
results = {}

# Define the output folder
output_folder = "stock_results"
os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

for ticker in tickers:
    print(f"Running model for {ticker}...\n")

    # Use the TechnicalIndicators class to get the stock data and technical indicators
    tech_indicators = TechnicalIndicators(ticker=ticker)  # Load data using ticker
    df_with_indicators = tech_indicators.add_technical_indicators()

    # Normalize the data (indicators)
    df, all_scaled_data = normalize_data(df_with_indicators)

    output_file = os.path.join(output_folder, f"{ticker}_results_LSTM.csv")

    with open(output_file, mode='w', newline='') as file:
        header = ["Ticker", "Indicators", "Units", "Dropout Rate", "Learning Rate","Batch Size", "Epochs", "Optimizer", "Activation", "Sequence Length","RMSE", "MAPE", "R2", "Accuracy"]
        writer = csv.writer(file)
        writer.writerow(header)  # Write header

        for hyperparam_values in all_param_combinations:
            params = dict(zip(param_keys, hyperparam_values))
            
            for selected_indicators, scaled_data in all_scaled_data.items():
                print(f"Training with indicators: {selected_indicators} and parameters: {params}")
                
                # Prepare sequence data with custom sequence_length
                sequence_length = params['sequence_length']
                X, y = create_time_series_data(df, scaled_data, window_size=sequence_length)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                # Scale y
                scaler_y = MinMaxScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
                y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

                input_shape = (X_train.shape[1], X_train.shape[2])
                model = create_lstm_model(input_shape, params)

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                history = model.fit(
                    X_train, y_train_scaled,
                    validation_data=(X_val, y_val_scaled),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )

                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                y_test_original = scaler_y.inverse_transform(y_test_scaled)

                mse = mean_squared_error(y_test_original, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100
                r2 = r2_score(y_test_original, y_pred)
                accuracy = calculate_accuracy(y_test_original, y_pred)

                writer.writerow([
                    ticker,
                    ', '.join(selected_indicators),
                    params['units'],
                    params['dropout_rate'],
                    params['learning_rate'],
                    params['batch_size'],
                    params['epochs'],
                    params['optimizer'],
                    params['activation'],
                    sequence_length,
                    rmse,
                    mape,
                    r2,
                    accuracy
                ])
                file.flush()


                print(f"{ticker} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%\n")
