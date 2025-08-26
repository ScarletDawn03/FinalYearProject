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

# Reproducibility settings
set_global_seed(seed=42, framework='torch')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Optuna Logging Setup ---
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------
# Model Definition
# -------------------------------
class SLSTMModel(nn.Module):
    def __init__(self, input_size, lstm_units, num_layers, dropout, dense_config, act_dense):
        super(SLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        activation_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        dense_layers, prev_units = [], lstm_units
        for i, units in enumerate(dense_config):
            dense_layers.append(nn.Linear(prev_units, units))
            if i < len(dense_config) - 1:
                dense_layers.append(activation_map[act_dense]())
            prev_units = units
        self.fc = nn.Sequential(*dense_layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# -------------------------------
# Training + Evaluation
# -------------------------------
def train_and_evaluate_model(
    model, train_loader, val_data, optimizer, criterion, scaler_y,
    forecast_window, epochs, patience=0.2, trial=None
):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_patience = int(patience * epochs) if isinstance(patience, float) else patience

    X_val_tensor = torch.tensor(val_data[0], dtype=torch.float32).to(device)
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Val Loss = {val_loss:.6f}")

        # ✅ Optuna pruning
        if trial is not None and (epoch + 1) % 5 == 0:
            trial.report(-val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= actual_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Final evaluation
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

    # ✅ Free GPU memory after each trial
    del X_val_tensor, y_val_scaled_tensor
    torch.cuda.empty_cache()

    return rmse, mape, r2, acc, profit_index


# -------------------------------
# Dataset Caching Utility
# -------------------------------
def prepare_and_cache_data(df, selected_features, window_size, forecast_window, ticker):
    cache_dir = f"cache/{ticker}_w{window_size}_f{forecast_window}"
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": f"{cache_dir}/X_train.npy",
        "X_val": f"{cache_dir}/X_val.npy",
        "y_train": f"{cache_dir}/y_train.npy",
        "y_val": f"{cache_dir}/y_val.npy",
        "scaler": f"{cache_dir}/scaler.npy"
    }

    if all(os.path.exists(p) for p in paths.values()):
        print(f"✅ Loaded cached dataset for (w={window_size}, f={forecast_window})")
        scaler_y = np.load(paths["scaler"], allow_pickle=True).item()
        return {**paths, "scaler_y": scaler_y}

    print(f"⚙️ Generating dataset for (w={window_size}, f={forecast_window})...")
    processor = DataPreprocessing(ticker=ticker)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])

    X, y = processor.create_windowed_data(scaled_data, window_size, forecast_window)
    X_train, X_val, _, y_train, y_val, _ = processor.split_dataset(X, y)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    np.save(paths["X_train"], X_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_train"], y_train_scaled)
    np.save(paths["y_val"], y_val_scaled)
    np.save(paths["scaler"], scaler_y, allow_pickle=True)

    return {**paths, "scaler_y": scaler_y}


# -------------------------------
# Optuna Objective
# -------------------------------
def objective(trial, cached_data, selected_indicators, ticker, window_size, forecast_window):
    X_train = np.load(cached_data["X_train"])
    X_val = np.load(cached_data["X_val"])
    y_train_scaled = np.load(cached_data["y_train"])
    y_val_scaled = np.load(cached_data["y_val"])
    scaler_y = cached_data["scaler_y"]

    hyperparams = {
        "lstm_units": trial.suggest_categorical("lstm_units", [16, 32, 64]),
        "num_layers": 2,
        "dropout": trial.suggest_float("dropout", 0.2, 0.5, step=0.1),
        "lr": trial.suggest_categorical("lr", [0.0001, 0.0005, 0.001]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "dense_config": trial.suggest_categorical("dense_config", [(16, 1), (25, 1)]),
        "act_dense": trial.suggest_categorical("act_dense", ["ReLU", "Tanh"]),
        "epochs": trial.suggest_categorical("epochs", [30,60,90])
    }

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        pin_memory=False,        # ⚠ ROCm safer without pinning
        num_workers=0            # ⚠ Start safe for ROCm
    )

    model = SLSTMModel(
        input_size=X_train.shape[2],
        lstm_units=hyperparams["lstm_units"],
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
        dense_config=hyperparams["dense_config"],
        act_dense=hyperparams["act_dense"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = nn.MSELoss()

    rmse, mape, r2, acc, profit_index = train_and_evaluate_model(
        model, train_loader, (X_val, y_val_scaled), optimizer, criterion,
        scaler_y, forecast_window, hyperparams["epochs"], trial=trial
    )

    with open(f'stock_results/{ticker}_SLSTM_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, ', '.join(selected_indicators),
            hyperparams["lstm_units"], hyperparams["num_layers"], hyperparams["dropout"],
            hyperparams["lr"], hyperparams["batch_size"], window_size, forecast_window,
            hyperparams["epochs"], hyperparams["dense_config"], hyperparams["act_dense"],
            rmse, mape, r2, acc, profit_index
        ])

    return -rmse if not np.isnan(rmse) and not np.isinf(rmse) else -1e10


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    ticker = 'BK'
    os.makedirs('stock_results', exist_ok=True)

    if not os.path.exists(f'stock_results/{ticker}_SLSTM_results.csv'):
        with open(f'stock_results/{ticker}_SLSTM_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Indicators", "LSTM Units", "Layers", "Dropout", "LR", "Batch Size",
                "Window Size", "Forecast Window", "Epochs", "Dense Config", "Dense Activation",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index"
            ])

    processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = processor.add_technical_indicators()

    selected_features = ['Close', '20MA', '50MA', 'RSI', 'MACD', 'Upper_BB',
                         'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV']

    all_combinations = list(combinations(selected_features[1:], 6))
    window_forecast_combos = [(60, 1), (60, 30)]

    for i, indicator_combo in enumerate(all_combinations):
        for j, (w, f) in enumerate(window_forecast_combos):
            print(f"\n=== [{i+1}/{len(all_combinations)}] Combo: {indicator_combo}, (w={w}, f={f}) ===")

            cached_data = prepare_and_cache_data(
                df_with_all_indicators,
                ['Close'] + list(indicator_combo),
                w, f, ticker
            )

            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(
                partial(objective, cached_data=cached_data, selected_indicators=indicator_combo,
                        ticker=ticker, window_size=w, forecast_window=f),
                n_trials=25
            )

            print(f"  -> Best score: {study.best_trial.value:.4f}")

    record_best_models(f'stock_results/{ticker}_SLSTM_results.csv')
