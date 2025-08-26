import os
import csv
import numpy as np
import optuna
import logging
import sys
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from functools import partial

# --- Reusable Functions ---
from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM
from reproducibility_settings import set_global_seed
from ReusableFunctions.RecordBestModel import record_best_models

# Reproducibility
set_global_seed(seed=42, framework='numpy')

# --- Optuna Logging ---
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.INFO)


def train_and_evaluate_rf(
    model: RandomForestRegressor,
    X_train, y_train, X_val, y_val, scaler_y, forecast_window
):
    model.fit(X_train, y_train.ravel())
    val_preds_scaled = model.predict(X_val).reshape(-1, 1)

    # Unscale predictions and targets
    val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled)
    y_val_true_unscaled = scaler_y.inverse_transform(y_val)

    # Metrics
    r2 = EM.r2(y_val_true_unscaled, val_preds_unscaled)
    if np.isnan(r2) or np.isinf(r2):
        rmse, mape, acc = np.nan, np.nan, np.nan
    else:
        rmse = EM.rmse(y_val_true_unscaled, val_preds_unscaled)
        mape = EM.mape(y_val_true_unscaled, val_preds_unscaled)
        acc = EM.accuracy(y_val_true_unscaled, val_preds_unscaled)

    profit_index = EM.profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window)
    return rmse, mape, r2, acc, profit_index


def objective(trial, df, selected_indicators, ticker, window_size, forecast_window):
    selected_features = ['Close'] + list(selected_indicators)
    data_processor = DataPreprocessing(ticker=ticker, start_date='2014-01-01', end_date='2024-12-31')
    df = data_processor.add_technical_indicators()

    # Scale the selected features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])

    # Generate windowed data
    X, y = data_processor.create_windowed_data(scaled_data, window_size, forecast_window)
    X_train, X_val, _, y_train, y_val, _ = data_processor.split_dataset(X, y)

    # Scale target
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Random Forest hyperparameters
    hyperparams = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = RandomForestRegressor(**hyperparams, random_state=42, n_jobs=-1)

    rmse, mape, r2, acc, profit_index = train_and_evaluate_rf(
        model, X_train.reshape(X_train.shape[0], -1), y_train_scaled,
        X_val.reshape(X_val.shape[0], -1), y_val_scaled, scaler_y, forecast_window
    )

    # Save results
    with open(f'stock_results/{ticker}_RF_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, ', '.join(selected_indicators),
            hyperparams["n_estimators"], hyperparams["max_depth"], hyperparams["min_samples_split"],
            hyperparams["min_samples_leaf"], hyperparams["max_features"],
            window_size, forecast_window,
            rmse, mape, r2, acc, profit_index
        ])

    
    return -rmse if not np.isnan(rmse) and not np.isinf(rmse) else -1e10

if __name__ == '__main__':
    ticker = 'C'
    os.makedirs('stock_results', exist_ok=True)
    write_header = not os.path.exists(f'stock_results/{ticker}_RF_results.csv')

    if write_header:
        with open(f'stock_results/{ticker}_RF_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Indicators", "n_estimators", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features",
                "Window Size", "Forecast Window",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index"
            ])

    data_processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = data_processor.add_technical_indicators()
    os.makedirs('check', exist_ok=True)
    df_with_all_indicators.to_csv(f'check/{ticker}_RF_downloaded_data.csv', index=True)
    print(f"Downloaded and processed data saved to check/{ticker}_RF_downloaded_data.csv")

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
            print(f"  -> Best R2 for {indicator_combo} with (w={window_size}, f={forecast_window}): {study.best_trial.value:.4f}")

    record_best_models(f'stock_results/{ticker}_RF_results.csv')
