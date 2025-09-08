import os
import csv
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM
from ReusableFunctions.RecordBestModel import record_best_models

# -------------------------------
# Training + Evaluation
# -------------------------------
def train_and_evaluate_linear_model(X_train, y_train, X_val, y_val, scaler_y, forecast_window):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # Ordinary Least Squares Linear Regression
    model = LinearRegression(fit_intercept=True)

    model.fit(X_train_flat, y_train)
    val_preds_scaled = model.predict(X_val_flat)
    val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1))
    y_val_true_unscaled = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    r2 = EM.r2(y_val_true_unscaled, val_preds_unscaled)
    rmse = EM.rmse(y_val_true_unscaled, val_preds_unscaled)
    mape = EM.mape(y_val_true_unscaled, val_preds_unscaled)
    acc = EM.accuracy(y_val_true_unscaled, val_preds_unscaled)
    profit_index = EM.profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window)

    return rmse, mape, r2, acc, profit_index

# -------------------------------
# Dataset Caching
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
    processor = DataPreprocessing(df=df)
    scaler_y = MinMaxScaler()

    scaled_data = MinMaxScaler().fit_transform(df[selected_features])
    X, y = processor.create_windowed_data(scaled_data, window_size, forecast_window)
    X_train, X_val, _, y_train, y_val, _ = processor.split_dataset(X, y)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    np.save(paths["X_train"], X_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_train"], y_train_scaled)
    np.save(paths["y_val"], y_val_scaled)
    np.save(paths["scaler"], scaler_y, allow_pickle=True)

    return {**paths, "scaler_y": scaler_y}

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    ticker = '5258.KL'
    os.makedirs('stock_results', exist_ok=True)

    result_file = f'stock_results/{ticker}_LinearModels_results.csv'
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Indicators", "Window Size", "Forecast Window",
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

            X_train = np.load(cached_data["X_train"])
            X_val = np.load(cached_data["X_val"])
            y_train_scaled = np.load(cached_data["y_train"])
            y_val_scaled = np.load(cached_data["y_val"])
            scaler_y = cached_data["scaler_y"]

            rmse, mape, r2, acc, profit_index = train_and_evaluate_linear_model(
                X_train, y_train_scaled, X_val, y_val_scaled,
                scaler_y, f
            )

            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ', '.join(indicator_combo),
                    w, f,
                    rmse, mape, r2, acc, profit_index
                ])

    record_best_models(result_file)
