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
import pandas as pd # <--- Make sure this is imported!

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

# --- Reusable Functions ---
from ReusableFunctions.DataPreprocessing import DataPreprocessing

class CNNModel(nn.Module):
    """
    A simple 1D CNN model for time series prediction.
    """
    def __init__(self, input_shape: tuple, filters: int, kernel_size: int, dropout_rate: float):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=filters, kernel_size=kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.dropout = nn.Dropout(dropout_rate)
        
        pooled_length = input_shape[1] // 4
        self.fc1 = nn.Linear(filters * pooled_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

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
    epochs: int = 20,
    patience: int = 5 # For early stopping
) -> tuple[float, float, float, float]: # Returns rmse, mape, r2, accuracy
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
    
    return rmse, mape, r2, acc

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    It evaluates different hyperparameter combinations and indicator sets.
    """
    ticker = 'AAPL'
    print(f"\n--- Starting Optuna Trial {trial.number} ---")

    data_processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = data_processor.add_technical_indicators()

    selected_base_indicators = [
        '20MA', '50MA', 'RSI', 'MACD', 'Upper_BB', 
        'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV'
    ]

    df, all_scaled_data = normalize_data(df_with_all_indicators, selected_base_indicators)

    hyperparams = {
        "filters": trial.suggest_categorical("filters", [32, 64, 128]),
        "kernel_size": trial.suggest_int("kernel_size", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "window_size": trial.suggest_categorical("window_size", [5, 60]),
        "forecast_window": trial.suggest_categorical("forecast_window", [1, 30])   
    }
    print(f"Trial {trial.number} Hyperparameters: {hyperparams}")

    out_dir = 'stock_results'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{ticker}_optuna_results_CNN_reduced_combinations.csv")
    
    write_header = not os.path.exists(csv_path)
    
    best_r2_for_trial = -np.inf

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Trial", "Indicators", "Filters", "Kernel Size", "Dropout", 
                "LR", "Batch Size", "Window Size", "Forecast Window",  # <-- add
                "RMSE", "MAPE", "R2", "Accuracy"
            ])

        total_combinations = len(all_scaled_data)
        for i, (current_selected_indicators, (scaled_data, _)) in enumerate(all_scaled_data.items()):
            # Changed this print statement for better clarity with the epoch prints below
            print(f"\n  Trial {trial.number}: Combination {i+1}/{total_combinations} - Indicators: {', '.join(current_selected_indicators)}")
            print(f"  -------------------------------------------------------------")

            try:
                X, y = create_time_series_data(
                    df_with_all_indicators, scaled_data,
                    window_size=hyperparams["window_size"],
                    forecast_window=hyperparams["forecast_window"]
                )
                X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

                scaler_y = MinMaxScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
                y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
                
                train_loader = DataLoader(
                    TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
                                  torch.tensor(y_train_scaled, dtype=torch.float32)),
                    batch_size=hyperparams["batch_size"],
                    shuffle=True
                )

                model = CNNModel(
                    input_shape=(X_train.shape[2], X_train.shape[1]), 
                    filters=hyperparams["filters"],
                    kernel_size=hyperparams["kernel_size"],
                    dropout_rate=hyperparams["dropout"]
                ).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
                criterion = nn.MSELoss()

                rmse, mape, r2, acc = train_and_evaluate_model(
                    model,
                    train_loader,
                    (X_val, y_val_scaled),
                    optimizer,
                    criterion,
                    scaler_y
                )
                
                print(f"  -------------------------------------------------------------")
                print(f"  Combination {i+1}/{total_combinations} Metrics: RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}, Accuracy={acc:.2f}%")
                print(f"  -------------------------------------------------------------")

                writer.writerow([
                    trial.number,
                    ', '.join(current_selected_indicators),
                    hyperparams["filters"], hyperparams["kernel_size"], hyperparams["dropout"], 
                    hyperparams["lr"], hyperparams["batch_size"],
                    hyperparams["window_size"], hyperparams["forecast_window"],  
                    rmse, mape, r2, acc
                ])

                if r2 > best_r2_for_trial:
                    best_r2_for_trial = r2

            except Exception as e:
                print(f"  -------------------------------------------------------------")
                print(f"  ERROR on combination {', '.join(current_selected_indicators)}: {e}. Logging error and continuing.")
                print(f"  -------------------------------------------------------------")
                writer.writerow([
                    trial.number,
                    ', '.join(current_selected_indicators),
                    hyperparams["filters"], hyperparams["kernel_size"], hyperparams["dropout"], 
                    hyperparams["lr"], hyperparams["batch_size"],
                    hyperparams["window_size"], hyperparams["forecast_window"],  
                    "ERROR", str(e), np.nan, np.nan
                ])
                continue

    return best_r2_for_trial if best_r2_for_trial > -np.inf else -1.0

# --- Main Execution ---
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) 
    
    print("\n--- Optimization Finished ---")
    print("Best overall trial (based on best R2 found within a trial):")
    trial = study.best_trial
    print(f"  Value (Best R2 in a trial): {trial.value:.4f}")
    print("  Optimal Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")