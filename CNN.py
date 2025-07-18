import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import optuna
import logging
import sys
import pandas as pd 
from functools import partial
# --- Reusable Functions ---
from ReusableFunctions.DataPreprocessing import DataPreprocessing


# Reproducibility settings
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Optuna Logging Setup ---
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.INFO)


class CNNModel(nn.Module):
    def __init__(self, input_shape: tuple, filters: int, kernel_size: int, dropout_rate: float, activation: str):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=filters, kernel_size=kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.dropout = nn.Dropout(dropout_rate)

        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        pooled_length = input_shape[1] // 4
        self.fc1 = nn.Linear(filters * pooled_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def calculate_profitability_index(y_true: np.ndarray, y_pred: np.ndarray, forecast_window: int, initial_capital: float = 10000.0):
    """
    Calculates the profitability index as Final Capital / Initial Capital based on a simple trading strategy.
    - For forecast_window=1: Buy if predicted price > current price, sell otherwise.
    - For forecast_window=30: Buy if predicted price > current price, sell otherwise.
    """
    capital = initial_capital
    position = 0  # number of shares held
    stock_price_history = y_true.flatten()
    predicted_prices = y_pred.flatten()

    for i in range(len(predicted_prices) - forecast_window):
        current_price = stock_price_history[i]
        predicted_future_price = predicted_prices[i]

        if forecast_window == 1:
            if predicted_future_price > current_price and position == 0:
                position = capital / current_price  # buy
                capital = 0
            elif predicted_future_price < current_price and position > 0:
                capital = position * current_price  # sell
                position = 0

        elif forecast_window == 30:
            if predicted_future_price > current_price and position == 0:
                position = capital / current_price  # buy
                capital = 0
            elif predicted_future_price < current_price and position > 0:
                capital = position * current_price  # sell
                position = 0

    # Final liquidation if still holding stock
    if position > 0:
        capital = position * stock_price_history[-1]

    return capital / initial_capital  # Profitability Index



def normalize_data(df: pd.DataFrame, all_indicators: list[str]):
    """
    Normalizes the data for all possible indicator combinations from the provided list.
    Returns a dictionary mapping combinations to scaled data and their scalers.
    """
    indicator_combinations = list(combinations(all_indicators, 5)) 
    
    all_scaled_data = {}
    for selected_indicators in indicator_combinations:
        selected_features = ['Close'] + list(selected_indicators)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[selected_features])
        all_scaled_data[selected_indicators] = (scaled_data, scaler)
    return df, all_scaled_data

def create_time_series_data(df, scaled_data, window_size: int = 50, forecast_window: int = 1):
    """
    Creates time series data (X, y) from the dataframe and scaled data.
    X will be (num_samples, window_size, num_features)
    y will be (num_samples,)
    """
    X, y = [], []
    for i in range(window_size, len(df) - forecast_window + 1):
        X.append(scaled_data[i - window_size:i])
        y.append(df['Close'].iloc[i + forecast_window - 1])  # predict t+forecast_window
    return np.array(X), np.array(y)


def split_data(X: np.ndarray, y: np.ndarray, train_size: float = 0.75, val_size: float = 0.05):
    """Splits data into training, validation, and test sets."""
    total_samples = len(X)
    train_end = int(total_samples * train_size)
    val_end = train_end + int(total_samples * val_size)
    
    return (
        X[:train_end], X[train_end:val_end], X[val_end:],
        y[:train_end], y[train_end:val_end], y[val_end:]
    )

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold_percent: float = 5):
    """Calculates accuracy based on a percentage threshold."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_true = np.where(y_true == 0, 1e-8, y_true)
    percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
    return (np.sum(percentage_diff <= threshold_percent) / len(y_true)) * 100

def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_data: tuple[np.ndarray, np.ndarray], # (X_val, y_val_scaled)
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler_y: MinMaxScaler,
    forecast_window: int,
    epochs: int,
    patience: int = 10 # For early stopping
) -> tuple[float, float, float, float, float]: # Returns rmse, mape, r2, accuracy
    """
    Trains the CNN model and evaluates it on the validation set.
    Includes basic early stopping and prints epoch progress.
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0

    X_val_tensor = torch.tensor(val_data[0], dtype=torch.float32).permute(0, 2, 1).to(device)
    y_val_scaled_tensor = torch.tensor(val_data[1], dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        # Optional: Print batch progress here if individual batches are very long
        # print(f"        Epoch {epoch + 1}/{epochs} - Batch progress: ", end='')
        # for i, (xb, yb) in enumerate(train_loader):
        #     # ... training code ...
        #     if i % 10 == 0: print('.', end='') # Print a dot for every 10 batches
        # print() # Newline after batch progress

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_preds_scaled = model(X_val_tensor).squeeze()
            val_loss = criterion(val_preds_scaled, y_val_scaled_tensor.squeeze()).item()

            # Print epoch status with validation loss
            print(f"        Epoch {epoch + 1}/{epochs}: Val Loss = {val_loss:.6f}", end='')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(" (New best!)")
            else:
                epochs_no_improve += 1
                print(f" (No improvement for {epochs_no_improve}/{patience} epochs)")
                if epochs_no_improve >= patience: # Changed to >= for consistency
                    print(f"        Early stopping triggered at epoch {epoch+1}.")
                    break # Stop training

    # Final evaluation after training (or early stopping)
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_tensor).cpu().numpy()
        val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1))
        y_val_true_unscaled = scaler_y.inverse_transform(val_data[1])

    # Calculate metrics
    r2 = r2_score(y_val_true_unscaled, val_preds_unscaled)
    if np.isnan(r2) or np.isinf(r2):
        rmse, mape, acc = np.nan, np.nan, np.nan
    else:
        rmse = np.sqrt(mean_squared_error(y_val_true_unscaled, val_preds_unscaled))
        mape = mean_absolute_percentage_error(y_val_true_unscaled, val_preds_unscaled) * 100
        acc = calculate_accuracy(y_val_true_unscaled, val_preds_unscaled)

    # Profitability Index
    profitability = calculate_profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window=forecast_window)

    
    return rmse, mape, r2, acc, profitability


def objective(trial, df, selected_indicators, ticker, window_size, forecast_window, epochs):


    selected_features = ['Close'] + list(selected_indicators)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])

    hyperparams = {
        "filters": trial.suggest_categorical("filters", [32, 64, 128]),
        "kernel_size": trial.suggest_int("kernel_size", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5, step=0.1),
        "lr": trial.suggest_float("lr", 0.0001, 0.001, step=0.0001),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "window_size": window_size, 
        "forecast_window": forecast_window,
        "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu"]),
    }

    X, y = create_time_series_data(df, scaled_data, hyperparams["window_size"], hyperparams["forecast_window"])
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
                      torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=hyperparams["batch_size"], shuffle=True
    )

    model = CNNModel(input_shape=(X_train.shape[2], X_train.shape[1]),
                     filters=hyperparams["filters"],
                     kernel_size=hyperparams["kernel_size"],
                     dropout_rate=hyperparams["dropout"],
                     activation=hyperparams["activation"]
                     ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = nn.MSELoss()

    rmse, mape, r2, acc, profit_index= train_and_evaluate_model(
        model, train_loader, (X_val, y_val_scaled), optimizer, criterion, scaler_y, forecast_window, epochs=epochs
    )

    # Log results
    with open(f'stock_results/{ticker}_CNN_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, ', '.join(selected_indicators),
            hyperparams["filters"], hyperparams["kernel_size"], hyperparams["dropout"],
            hyperparams["lr"], hyperparams["batch_size"],
            hyperparams["window_size"], hyperparams["forecast_window"],epochs,
            hyperparams["activation"],
            rmse, mape, r2, acc, profit_index
        ])

    return -rmse if not np.isnan(rmse) and not np.isinf(rmse) else -1e10

# Main execution block
if __name__ == '__main__':
    ticker = 'AAPL'
    os.makedirs('stock_results', exist_ok=True)
    write_header = not os.path.exists(f'stock_results/{ticker}_CNN_results.csv')

    if write_header:
        with open(f'stock_results/{ticker}_CNN_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Indicators", "Filters", "Kernel Size", "Dropout",
                "LR", "Batch Size", "Window Size", "Forecast Window", "Epochs", "Activation Func",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index"
            ])

    data_processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = data_processor.add_technical_indicators()

    selected_base_indicators = [
        '20MA', '50MA', 'RSI', 'MACD', 'Upper_BB',
        'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV'
    ]
    all_combinations = list(combinations(selected_base_indicators, 5))
    window_forecast_combos = [(5, 1), (5, 30), (60, 1), (60, 30)]


    for i, indicator_combo in enumerate(all_combinations):
     for j, (window_size, forecast_window) in enumerate(window_forecast_combos):
      for epochs in [50, 100, 150]:
        print(f"\n=== [{i+1}/{len(all_combinations)}] Combo: {indicator_combo}")
        print(f"    -> Window Size: {window_size}, Forecast Window: {forecast_window} ===")

        study = optuna.create_study(direction='maximize')

        study.optimize(
            partial(
                objective,
                df=df_with_all_indicators,
                selected_indicators=indicator_combo,
                ticker=ticker,
                window_size=window_size,
                forecast_window=forecast_window,
                epochs=epochs
                
            ),
            n_trials=25
        )

        print(f"  -> Best R2 for indicators {indicator_combo} with (w={window_size}, f={forecast_window}): {study.best_trial.value:.4f}")
