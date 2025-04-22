import os
import sys

# Add the parent directory (where ReusableFunctions is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import numpy as np

from itertools import combinations, product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ReusableFunctions.DataPreprocessing import DataPreprocessing

# Define hyperparameter search space
hyperparameter_space = {
    'window_sizes': [200],
    'filters': [32, 64, 128],
    'pool_sizes': [2, 3, 4],
    'activations': ['relu', 'leaky_relu'],
    'epochs': [50,100,150,200]
}

# Fixed params
fixed_params = {
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 16,
    'optimizer': 'adam'
}

# Normalize data
def normalize_data(df):
    all_indicators = ['20MA', '50MA', '200MA', 'RSI', 'MACD', 'Signal_Line',
                      'Upper_BB', 'Lower_BB', 'CCI', 'ATR', 'ROC', 'Williams_%R', 'OBV']
    indicator_combinations = list(combinations(all_indicators, 5))
    results = {}
    all_scaled_data = {}
    for selected_indicators in indicator_combinations:
        selected_features = ['Close'] + list(selected_indicators)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[selected_features])
        all_scaled_data[selected_indicators] = scaled_data
    return df, all_scaled_data

# Time-series data
def create_time_series_data(df, scaled_data, window_size):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

# Data splitting
def split_data(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]

# CNN model creation
def create_cnn_model(input_shape, filters, kernel_size, pool_size, activation, dropout_rate, optimizer, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape))

    # Add activation
    if activation == 'relu':
        model.add(tf.keras.layers.Activation('relu'))
    elif activation == 'leaky_relu':
        model.add(LeakyReLU(alpha=0.01))

    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=1))

    # Optimizer
    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# Run models
tickers = ['AAPL']
output_folder = "stock_results"
os.makedirs(output_folder, exist_ok=True)

for ticker in tickers:
    print(f"Running model for {ticker}...\n")

    data_preprocessing = DataPreprocessing(ticker=ticker)
    df_with_indicators = data_preprocessing.add_technical_indicators()
    df, all_scaled_data = normalize_data(df_with_indicators)

    output_file = os.path.join(output_folder, f"{ticker}_results_CNN_GridSearch.csv")
    with open(output_file, mode='w', newline='') as file:
        header = ["Ticker", "Indicators", "Window Size", "Filters", "Kernel Size", "Pooling Size",
                  "Activation", "Dropout Rate", "Learning Rate", "Batch Size", "Epochs", "Optimizer",
                  "MSE", "RMSE", "MAE", "MAPE"]
        writer = csv.writer(file)
        writer.writerow(header)

        for selected_indicators, scaled_data in all_scaled_data.items():
            for window_size, filters, pool_size, activation, epochs in product(
                hyperparameter_space['window_sizes'],
                hyperparameter_space['filters'],
                hyperparameter_space['pool_sizes'],
                hyperparameter_space['activations'],
                hyperparameter_space['epochs']
            ):
                print(f"Training {ticker} with indicators: {selected_indicators}, window={window_size}, filters={filters}, pool={pool_size}, activation={activation}, epochs={epochs}")

                try:
                    X, y = create_time_series_data(df, scaled_data, window_size)
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                    scaler_y = MinMaxScaler()
                    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
                    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
                    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

                    input_shape = (X_train.shape[1], X_train.shape[2])

                    model = create_cnn_model(
                        input_shape=input_shape,
                        filters=filters,
                        kernel_size=3,
                        pool_size=pool_size,
                        activation=activation,
                        dropout_rate=fixed_params['dropout_rate'],
                        optimizer=fixed_params['optimizer'],
                        learning_rate=fixed_params['learning_rate']
                    )

                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    model.fit(
                        X_train, y_train_scaled,
                        epochs=epochs,
                        batch_size=fixed_params['batch_size'],
                        validation_data=(X_val, y_val_scaled),
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    y_pred_scaled = model.predict(X_test)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                    y_test_original = scaler_y.inverse_transform(y_test_scaled)

                    mse = mean_squared_error(y_test_original, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_original, y_pred)
                    mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100

                    writer.writerow([
                        ticker, ', '.join(selected_indicators), window_size, filters, 3, pool_size, activation,
                        fixed_params['dropout_rate'], fixed_params['learning_rate'], fixed_params['batch_size'],
                        epochs, fixed_params['optimizer'], mse, rmse, mae, mape
                    ])
                    file.flush()

                    print(f"✅ Done | RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%\n")

                except Exception as e:
                    print(f"Skipping combination due to error: {e}")
