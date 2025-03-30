#Importing the required libraries
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

#Error Supression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


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

# Adding technical indicators, customization for technical indicator, compare accuracy and pick higher accuracy 
def add_technical_indicators(ticker,df):
    df['20MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['50MA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['200MA'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    df['12EMA'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['26EMA'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = df['12EMA'] - df['26EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)

     # Verify no NaN values exist
    print("Null values in each column:\n", df.isnull().sum())
    print(f"Does the dataset contain any null values? {df.isnull().values.any()}")

    filename=f"{ticker}.csv"
    df.to_csv(filename)

    return df

# Normalize the data
def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close', '20MA', '50MA', '200MA', 'RSI', 'MACD']])
    return scaled_data, scaler

# Prepare the time-series data
def create_time_series_data(df, scaled_data, window_size=120):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

# Split data into training and testing sets (80% training,20% testing)
def split_data(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# Create RNN model
def create_rnn_model(input_shape):
    model = Sequential()
    #Takes in time series data
    model.add(Input(shape=input_shape))
    #RNN Layers
    model.add(SimpleRNN(units=50, return_sequences=True))
    model.add(SimpleRNN(units=50, return_sequences=False))
    #Dropout to reduce overfitting
    model.add(Dropout(0.2))
    #Optimizer
    model.add(Dense(units=1))
    model.add(LeakyReLU(alpha=0.1))  # Apply LeakyReLU

    #Loss Function
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

# Example usage
ticker = '5197.KL'
df = download_stock_data(ticker)
df = add_technical_indicators(ticker,df)

# print(data_check(df))

scaled_data, scaler = normalize_data(df)
X, y = create_time_series_data(df, scaled_data)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Scale target values (y)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

input_shape = (X_train.shape[1], X_train.shape[2])
model = create_rnn_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train_scaled,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val_scaled),  # Now using validation set
    callbacks=[early_stopping]
)


loss = model.evaluate(X_test, y_test_scaled)
print(f"Test loss: {loss}")

predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test_scaled)


#Compute Metrics
mse=mean_squared_error(y_test_original,predictions)
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
mae =mean_absolute_error(y_test_original,predictions)
mape=mean_absolute_percentage_error(y_test_original,predictions)
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Mean Square Error (MSE):{mse:.4f}")
print(f"Mean Absolute Error (MSE):{mae:.4f}")
print(f"Mean Absolute Percentagae Error (MAPE):{mape:.4f}")


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label="Actual Prices", color="blue", linewidth=2)
plt.plot(predictions, label="Predicted Prices", color="red", linestyle="dashed", linewidth=2)
plt.title(f"{ticker} Stock Price Prediction (RNN)")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()