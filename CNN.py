import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import optuna
import logging
import sys
import pandas as pd 
from functools import partial

# --- Reusable Functions ---
from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM
from reproducibility_settings import set_global_seed
from ReusableFunctions.RecordBestModel import record_best_models  # if you’ve saved it here


# Reproducibility settings
set_global_seed(seed=42, framework='torch')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# --- Optuna Logging Setup ---
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.INFO)


class CNNModel(nn.Module):
    def __init__(self, input_shape: tuple, filters: int, kernel_size: int, pooling_size: int, dropout_rate: float, activation: str):
        super(CNNModel, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=filters, kernel_size=kernel_size, padding=padding)

        self.pool = nn.MaxPool1d(kernel_size=pooling_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Dynamically determine output shape after conv and pool
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1])
            x = self.pool(self.activation(self.conv1(dummy_input)))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_data: tuple[np.ndarray, np.ndarray],  # (X_val, y_val_scaled)
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler_y: MinMaxScaler,
    forecast_window: int,
    epochs: int,
    patience: float = 0.2  # For early stopping
) -> tuple[float, float, float, float, float]:
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_patience = int(patience * epochs) if isinstance(patience, float) else patience


    print("Model is on device:", next(model.parameters()).device)

    X_val_tensor = torch.tensor(val_data[0], dtype=torch.float32).permute(0, 2, 1).to(device)
    y_val_scaled_tensor = torch.tensor(val_data[1], dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds_scaled = model(X_val_tensor).squeeze()
            val_loss = criterion(val_preds_scaled, y_val_scaled_tensor.squeeze()).item()

            print(f"        Epoch {epoch + 1}/{epochs}: Val Loss = {val_loss:.6f}", end='')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(" (New best!)")
            else:
                epochs_no_improve += 1
                print(f" (No improvement for {epochs_no_improve}/{patience} epochs)")
                if epochs_no_improve >= actual_patience:
                    print(f"        Early stopping triggered at epoch {epoch+1}.")
                    break

    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_tensor).cpu().numpy()
        val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1))
        y_val_true_unscaled = scaler_y.inverse_transform(val_data[1])

    # Use evaluation module
    r2 = EM.r2(y_val_true_unscaled, val_preds_unscaled)
    if np.isnan(r2) or np.isinf(r2):
        rmse, mape, acc = np.nan, np.nan, np.nan
    else:
        rmse = EM.rmse(y_val_true_unscaled, val_preds_unscaled)
        mape = EM.mape(y_val_true_unscaled, val_preds_unscaled)
        acc = EM.accuracy(y_val_true_unscaled, val_preds_unscaled)

    profit_index = EM.profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window=forecast_window)

    return rmse, mape, r2, acc, profit_index


def objective(trial, df, selected_indicators, ticker, window_size, forecast_window):
    selected_features = ['Close'] + list(selected_indicators)
    data_processor = DataPreprocessing(ticker=ticker, start_date='2014-01-01', end_date='2024-12-31')
    df = data_processor.add_technical_indicators()  # Apply indicators after removing noise


    # Scale the selected features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])

    # Generate windowed data using class method
    X, y = data_processor.create_windowed_data(scaled_data, window_size, forecast_window)

    # Split the data
    X_train, X_val, _, y_train, y_val, _ = data_processor.split_dataset(X, y)

    # Scale the targets
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Define hyperparameters
    hyperparams = {
        "filters": trial.suggest_categorical("filters", [32, 64, 128]),
        "kernel_size": trial.suggest_int("kernel_size", 2, 3),
        "pooling_size": trial.suggest_int("pooling_size", 2, 3),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5, step=0.1),
        "lr": trial.suggest_categorical("lr", [0.0001, 0.0005, 0.001]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "window_size": window_size,
        "forecast_window": forecast_window,
        "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu"]),   
    }

    epochs = trial.suggest_categorical("epochs", [50, 100, 150])

    # Create data loader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
                      torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=hyperparams["batch_size"], shuffle=False
    )

    # Build model
    model = CNNModel(
        input_shape=(X_train.shape[2], X_train.shape[1]),
        filters=hyperparams["filters"],
        kernel_size=hyperparams["kernel_size"],
        pooling_size=hyperparams["pooling_size"],
        dropout_rate=hyperparams["dropout"],
        activation=hyperparams["activation"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = nn.MSELoss()

    # Train and evaluate
    rmse, mape, r2, acc, profit_index = train_and_evaluate_model(
        model, train_loader, (X_val, y_val_scaled), optimizer, criterion,
        scaler_y, forecast_window, epochs
    )

    # Log results
    with open(f'stock_results/{ticker}_CNN_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, ', '.join(selected_indicators),
            hyperparams["filters"], hyperparams["kernel_size"], hyperparams["pooling_size"], hyperparams["dropout"],
            hyperparams["lr"], hyperparams["batch_size"],
            hyperparams["window_size"], hyperparams["forecast_window"], epochs,
            hyperparams["activation"],
            rmse, mape, r2, acc, profit_index
        ])

    return -rmse if not np.isnan(rmse) and not np.isinf(rmse) else -1e10


# Main execution block
if __name__ == '__main__':
    ticker = '1023.KL'
    os.makedirs('stock_results', exist_ok=True)
    write_header = not os.path.exists(f'stock_results/{ticker}_CNN_results.csv')

    if write_header:
        with open(f'stock_results/{ticker}_CNN_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Indicators", "Filters", "Kernel Size", "Pooling Size", "Dropout",
                "LR", "Batch Size", "Window Size", "Forecast Window", "Epochs", "Activation Func",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index"
            ])

    data_processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = data_processor.add_technical_indicators()
    # Save raw + indicator-enhanced data
    os.makedirs('check', exist_ok=True)
    df_with_all_indicators.to_csv(f'check/{ticker}_CNN_downloaded_data.csv', index=True)
    print(f"Downloaded and processed data saved to check/{ticker}_CNN_downloaded_data.csv")


    selected_base_indicators = [
        '20MA', '50MA', 'RSI', 'MACD', 'Upper_BB',
        'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV'
    ]
    all_combinations = list(combinations(selected_base_indicators, 6))
    window_forecast_combos = [(60, 1), (60, 30)]


    for i, indicator_combo in enumerate(all_combinations):
     for j, (window_size, forecast_window) in enumerate(window_forecast_combos):
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
            ),
            n_trials=25
        )

        print(f"  -> Best R2 for indicators {indicator_combo} with (w={window_size}, f={forecast_window}): {study.best_trial.value:.4f}")

 
    # After all Optuna trials have been completed
    csv_path = f'stock_results/{ticker}_CNN_results.csv'
    record_best_models(csv_path)

