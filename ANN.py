# Importing the required libraries
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

tf.config.threading.set_intra_op_parallelism_threads(8)  # Use 8 threads for intra-op parallelism
tf.config.threading.set_inter_op_parallelism_threads(8)  # Use 8 threads for inter-op paralleli
import csv

from itertools import combinations

# Error Suppression
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import ParameterGrid

# Define the hyperparameter grid
param_grid = {
    'hidden_units': [50, 100],  # Number of units in hidden layers
    'dropout_rate': [0.2, 0.3],  # Dropout rate
    'learning_rate': [0.001, 0.01],  # Learning rate
    'batch_size': [32, 64],  # Batch size
    'epochs': [20, 50],  # Number of epochs
    'optimizer': ['adam', 'rmsprop']  # Optimizer
}

# Convert the grid to a list of parameter combinations
param_combinations = list(ParameterGrid(param_grid))

# Function to download stock data
def download_stock_data(ticker, start_date='2010-01-01', end_date='2024-12-31'):
    # Calculate extra buffer period (e.g., 1 year)
    buffer_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    
    # Download data with buffer period
    data = yf.download(ticker, start=buffer_start.strftime('%Y-%m-%d'), end=end_date)

    # Fetch earnings report dates
    ticker_data = yf.Ticker(ticker)
    earnings_dates = set(pd.to_datetime(ticker_data.earnings_dates.index).normalize())  # Normalize dates

    # Fetch ex-dividend dates
    ex_dividend_dates = set(pd.to_datetime(ticker_data.dividends.index).normalize())

    # Combine both sets
    dates_to_remove = earnings_dates.union(ex_dividend_dates)

    # Ensure data index is datetime and normalized
    data.index = pd.to_datetime(data.index).normalize()

    # Remove both earnings & ex-dividend dates
    data = data[~data.index.isin(dates_to_remove)]

    # Trim the dataset to the actual requested start_date
    data = data.loc[start_date:]

    print(dates_to_remove)

    return data

# Function to add technical indicators
def add_technical_indicators(ticker, df):
    # Ensure calculations don't cause issues with chained assignments
    pd.options.mode.chained_assignment = None  

    # Moving Averages
    df['20MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['50MA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['200MA'] = df['Close'].rolling(window=200, min_periods=1).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)

    # MACD and Signal Line
    df['12EMA'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['26EMA'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = df['12EMA'] - df['26EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

    # Bollinger Bands
    df['20STD'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['Upper_BB'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_BB'] = df['20MA'] - (df['20STD'] * 2)

    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    mean_dev = lambda x: np.mean(np.abs(x - np.mean(x)))
    df['CCI'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / \
                (0.015 * typical_price.rolling(window=20, min_periods=1).apply(mean_dev, raw=True))

    # Average True Range (ATR)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
    df.drop(columns=['TR'], inplace=True)  # Drop intermediate column

    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    # Williams %R
    df['Williams_%R'] = ((df['High'].rolling(window=14, min_periods=1).max() - df['Close']) / 
                          (df['High'].rolling(window=14, min_periods=1).max() - df['Low'].rolling(window=14, min_periods=1).min())) * -100

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Drop intermediate columns
    df.drop(columns=['20STD'], inplace=True)

    # Forward-fill and backward-fill NaN values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Verify no NaN values exist
    print("Null values in each column:\n", df.isnull().sum())
    print(f"Does the dataset contain any null values? {df.isnull().values.any()}")

    folder_name = "check"
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f"{ticker}_ANN.csv")
    df.to_csv(filename, index=False)

    return df  # Original DataFrame is modified, so return is optional

# Function to normalize data
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

    return df, all_scaled_data  # Return the full dataset and all scaled variations

# Function to create time-series data
def create_time_series_data(df, scaled_data, window_size=120):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

# Function to split data into training and testing sets
def split_data(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to create an ANN model
def create_ann_model(input_shape, hidden_units=100, dropout_rate=0.2, learning_rate=0.001, optimizer='adam'):
    # Initialize the model
    model = Sequential()

    # Add Input layer explicitly
    model.add(Input(shape=input_shape))  # Explicitly define the input shape
    model.add(Flatten())  # Flatten the input data

    # First hidden layer
    model.add(Dense(units=hidden_units, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # Second hidden layer
    model.add(Dense(units=hidden_units, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    # Output layer
    model.add(Dense(units=1))  # Assuming you're predicting a single value

    # Choose optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer must be 'adam' or 'rmsprop'.")

    # Compile the model
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model

# Define tickers
tickers = ['AAPL', 'NVDA', 'MSFT', 'BRK-B', 'JPM', 'V', 'XOM', 'CVX', 'COP', '1155.KL', '1023.KL','1295.KL', '0270.KL', '5309.KL', '6963.KL', '5681.KL', '7727.KL', '7293.KL']

# Dictionary to store results
results = {}

# Define the output folder
output_folder = "stock_results"
os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

for ticker in tickers:
    print(f"Running model for {ticker}...\n")

    df = download_stock_data(ticker)
    df = add_technical_indicators(ticker, df)
    df, all_scaled_data = normalize_data(df)

    output_file = os.path.join(output_folder, f"{ticker}_results_ANN.csv")

    with open(output_file, mode='w', newline='') as file:
        header = ["Ticker", "Indicators", "Hidden Units", "Dropout Rate", "Learning Rate", "Batch Size", "Epochs", "Optimizer", "MSE", "RMSE", "MAE", "MAPE"]
        writer = csv.writer(file)
        writer.writerow(header)  # Write header

        for selected_indicators, scaled_data in all_scaled_data.items():
            for params in param_combinations:  # Loop through hyperparameter combinations
                print(f"Training with indicators: {selected_indicators} and parameters: {params}")
                X, y = create_time_series_data(df, scaled_data)

                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                # Scale target values (y)
                scaler_y = MinMaxScaler(feature_range=(0, 1))
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
                y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

                input_shape = (X_train.shape[1], X_train.shape[2])  # Input shape for ANN

                # Create and train the model with the current hyperparameters
                model = create_ann_model(
                    input_shape=input_shape,
                    hidden_units=params['hidden_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate'],
                    optimizer=params['optimizer']
                )

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = model.fit(
                    X_train, y_train_scaled,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(X_val, y_val_scaled),
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Evaluate the model
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                y_test_original = scaler_y.inverse_transform(y_test_scaled)

                mse = mean_squared_error(y_test_original, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_original, y_pred)
                mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100

                # Write the results to the CSV file
                writer.writerow([
                    ticker,  # Ticker symbol
                    ', '.join(selected_indicators),  # Selected indicators
                    params['hidden_units'],  # Hidden units
                    params['dropout_rate'],  # Dropout rate
                    params['learning_rate'],  # Learning rate
                    params['batch_size'],  # Batch size
                    params['epochs'],  # Number of epochs
                    params['optimizer'],  # Optimizer
                    mse,  # MSE
                    rmse,  # RMSE
                    mae,  # MAE
                    mape  # MAPE
                ])

                # Flush the file buffer to ensure data is written
                file.flush()

                print(f"{ticker} - RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}\n")