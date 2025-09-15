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
from functools import partial

# --- Reusable Functions ---
from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM
from reproducibility_settings import set_global_seed
from ReusableFunctions.RecordBestModel import record_best_models

# -------------------------------
# Reproducibility
# -------------------------------
set_global_seed(seed=42, framework='torch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

# -------------------------------
# Optuna Logging
# -------------------------------
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------
# LSTM Model
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, activation=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last hidden state
        return self.fc(out)

# -------------------------------
# Dataset Preparation (No Cache)
# -------------------------------
def prepare_data(df, selected_features, window_size, forecast_window, ticker):
    processor = DataPreprocessing(ticker=ticker)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])

    X, y = processor.create_windowed_data(scaled_data, window_size, forecast_window)
    X_train, X_val, _, y_train, y_val, _ = processor.split_dataset(X, y)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "scaler_y": scaler_y
    }

# -------------------------------
# Training + Evaluation
# -------------------------------
def train_and_evaluate_model(model, train_loader, val_data, optimizer, criterion, scaler_y,
                             forecast_window, epochs, patience=0.2):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_patience = int(patience * epochs) if isinstance(patience, float) else patience

    X_val_tensor = torch.tensor(val_data[0], dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(val_data[1], dtype=torch.float32).to(device)

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
            val_loss = criterion(val_preds_scaled, y_val_tensor.squeeze()).item()

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= actual_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_tensor).cpu().numpy()
        val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1))
        y_val_true_unscaled = scaler_y.inverse_transform(val_data[1])

    r2 = EM.r2(y_val_true_unscaled, val_preds_unscaled)
    rmse, mape, acc = (np.nan, np.nan, np.nan) if np.isnan(r2) or np.isinf(r2) else (
        EM.rmse(y_val_true_unscaled, val_preds_unscaled),
        EM.mape(y_val_true_unscaled, val_preds_unscaled),
        EM.accuracy(y_val_true_unscaled, val_preds_unscaled),
    )
    profit_index = EM.profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window)

    del X_val_tensor, y_val_tensor
    torch.cuda.empty_cache()
    return rmse, mape, r2, acc, profit_index

# -------------------------------
# Optuna Objective
# -------------------------------
def objective(trial, data, selected_indicators, ticker, window_size, forecast_window):
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train_scaled = data["y_train"]
    y_val_scaled = data["y_val"]
    scaler_y = data["scaler_y"]

    hyperparams = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.8, step=0.1),
        "lr": trial.suggest_categorical("lr", [0.0001, 0.0005, 0.001]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_categorical("epochs", [50, 100, 150])
    }

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=hyperparams["batch_size"],
        shuffle=False
    )

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=hyperparams["hidden_size"],
        num_layers=1,
        dropout=hyperparams["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = nn.MSELoss()

    rmse, mape, r2, acc, profit_index = train_and_evaluate_model(
        model, train_loader, (X_val, y_val_scaled), optimizer, criterion,
        scaler_y, forecast_window, hyperparams["epochs"]
    )

    # Log results
    with open(f'stock_results/{ticker}_LSTM_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, ', '.join(selected_indicators),
            hyperparams["hidden_size"], hyperparams["dropout"],
            hyperparams["lr"], hyperparams["batch_size"],
            window_size, forecast_window, hyperparams["epochs"],
            rmse, mape, r2, acc, profit_index
        ])

    return -rmse if not np.isnan(rmse) and not np.isinf(rmse) else -1e10

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    ticker = 'BK'
    os.makedirs('stock_results', exist_ok=True)
    if not os.path.exists(f'stock_results/{ticker}_LSTM_results.csv'):
        with open(f'stock_results/{ticker}_LSTM_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Indicators", "Hidden Size", "Dropout", "LR", "Batch Size",
                "Window Size", "Forecast Window", "Epochs",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index"
            ])

    data_processor = DataPreprocessing(ticker)
    df_with_all_indicators = data_processor.add_technical_indicators()
    os.makedirs('check', exist_ok=True)
    df_with_all_indicators.to_csv(f'check/{ticker}_LSTM_downloaded_data.csv', index=True)

    selected_base_indicators = ['20MA', '50MA', 'RSI', 'MACD', 'Upper_BB','Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV']
    all_combinations = list(combinations(selected_base_indicators, 6))
    window_forecast_combos = [(5, 1)]

    start_combo_index = 147 # Change to your desired start index
    start_config_index = 0 


    for i, indicator_combo in enumerate(all_combinations):

        if i < start_combo_index:
            continue

        for j, (window_size, forecast_window) in enumerate(window_forecast_combos):

            if i == start_combo_index and j < start_config_index:
                continue
             
            print(f"\n=== [{i+1}/{len(all_combinations)}] Combo: {indicator_combo}, (w={window_size}, f={forecast_window}) ===")
            data = prepare_data(
                df_with_all_indicators,
                ['Close'] + list(indicator_combo),
                window_size,
                forecast_window,
                ticker
            )

            # Optuna study without pruning
            study = optuna.create_study(direction='maximize')
            study.optimize(
                partial(objective,
                        data=data,
                        selected_indicators=indicator_combo,
                        ticker=ticker,
                        window_size=window_size,
                        forecast_window=forecast_window),
                n_trials=25
            )

            print(f"  -> Best RMSE: {-study.best_trial.value:.4f}")

    record_best_models(f'stock_results/{ticker}_LSTM_results.csv')
