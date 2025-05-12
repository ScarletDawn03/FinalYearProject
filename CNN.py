import os
import csv
import random
import numpy as np

from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,  r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

#Reproducibility settings
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Import custom class for technical indicators
from ReusableFunctions.DataPreprocessing import DataPreprocessing
# Fixed hyperparameters used across all models
fixed_params = {
    'filters': 32,  # Number of filters in the CNN layer
    'kernel_size': 3,  # Kernel size for convolution
    'dropout_rate': 0.2,  # Dropout rate to prevent overfitting
    'learning_rate': 0.001,  # Learning rate for optimizer
    'batch_size': 16, # Batch size during training
    'epochs': 100, # Maximum number of training epochs
    'optimizer': 'adam' # Optimizer
}

# Function to normalize the data
def normalize_data(df):

    #Normalize selected combinations of technical indicators using MinMaxScaler.
    #Returns the original dataframe and a dictionary of scaled datasets.
    
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

# Prepare the time-series data
def create_time_series_data(df, scaled_data, window_size=50):

    #  Convert scaled time-series data into input/output sequences for the model.

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i-window_size:i])
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)


def split_data(X, y, train_size=0.7, val_size=0.1):

    # Split data into training and testing sets (70% training,10% validation, 20% testing)

    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to create a CNN model with fixed hyperparameters
def create_cnn_model(input_shape, params=fixed_params):

    #Build and compile a 1D CNN model 

    # Initialize the model
    model = Sequential()

    # Add 1D convolutional layer
    model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], activation='relu', input_shape=input_shape))

    # Add MaxPooling layer
    model.add(MaxPooling1D(pool_size=4))

    # Dropout layer to prevent overfitting
    model.add(Dropout(rate=params['dropout_rate']))

    # Flatten the output to feed into fully connected layers
    model.add(Flatten())

    # Dense layer for final prediction
    model.add(Dense(units=1))

    # Choose optimizer
    if params['optimizer'] == 'adam':
        opt = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        opt = RMSprop(learning_rate=params['learning_rate'])
    else:
        raise ValueError("Optimizer must be 'adam' or 'rmsprop'.")

    # Compile the model
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model



def calculate_accuracy(y_true, y_pred, threshold_percent=5):

    # Calculate custom accuracy metric: % of predictions within threshold of actual value

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
    within_threshold = percentage_diff <= threshold_percent

    correct_count = np.sum(within_threshold)  # Count of correct predictions within threshold
    total_count = len(within_threshold)  # Total number of predictions
    accuracy_percentage = (correct_count / total_count) * 100  # Calculate accuracy percentage

    # Print for debugging or further analysis
    print(f"Correct predictions within {threshold_percent}%: {correct_count} out of {total_count}")
    return accuracy_percentage


# Define tickers
tickers = ['AAPL']

# Create output directory if it doesn't exist
output_folder = "stock_results"
os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

for ticker in tickers:
    print(f"Running model for {ticker}...\n")

    
    # Load stock data and technical indicators
    data_preprocessing = DataPreprocessing(ticker=ticker)  
    df_with_indicators = data_preprocessing.add_technical_indicators()
    
    # Save intermediate data for verification
    check_folder = "check"
    os.makedirs(check_folder, exist_ok=True)
    check_file_path = os.path.join(check_folder, f"{ticker}_CNN.csv")
    df_with_indicators.to_csv(check_file_path, index=False)


    # Normalize the data (indicators)
    df, all_scaled_data = normalize_data(df_with_indicators)

    # Prepare output file for logging results
    output_file = os.path.join(output_folder, f"{ticker}_results_CNN.csv")
    with open(output_file, mode='w', newline='') as file:
        header = ["Ticker", "Indicators", "Filters", "Kernel Size", "Dropout Rate", "Learning Rate", "Batch Size", "Epochs", "Optimizer",
          "RMSE", "MAPE", "R2", "Accuracy"]

        writer = csv.writer(file)
        writer.writerow(header)  # Write header

        for selected_indicators, scaled_data in all_scaled_data.items():
            print(f"Training with indicators: {selected_indicators} and fixed parameters: {fixed_params}")

            X, y = create_time_series_data(df, scaled_data)
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

             # Normalize target values (Close price)
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

            input_shape = (X_train.shape[1], X_train.shape[2])  # Input shape for CNN

            # Initialize and train the CNN model
            model = create_cnn_model(input_shape=input_shape)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                X_train, y_train_scaled,
                epochs=fixed_params['epochs'],
                batch_size=fixed_params['batch_size'],
                validation_data=(X_val, y_val_scaled),
                callbacks=[early_stopping],
                verbose=0
            )

            # Collect final training/validation loss for reporting
            # final_train_loss = history.history['loss'][-1]
            # final_val_loss = history.history['val_loss'][-1]


            # Predict and inverse scale predictions
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_test_original = scaler_y.inverse_transform(y_test_scaled)

            #Calculate Evaluation Metrics
            accuracy = calculate_accuracy(y_test_original, y_pred)
            mse=mean_squared_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100
            r2 = r2_score(y_test_original, y_pred)

            # Write the results to the CSV file
            writer.writerow([ticker, ', '.join(selected_indicators), fixed_params['filters'], fixed_params['kernel_size'], fixed_params['dropout_rate'],
                 fixed_params['learning_rate'], fixed_params['batch_size'], fixed_params['epochs'], fixed_params['optimizer'],
                 rmse, mape, r2, accuracy])
            
            # Can add in final_train_loss, final_val_loss, for FYP2


            # Flush the file buffer to ensure data is written
            file.flush()

            print(f"{ticker} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%\n")
