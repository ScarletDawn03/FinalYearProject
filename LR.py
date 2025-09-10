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
    """Train Linear Regression and evaluate on validation set."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

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

    return rmse, mape, r2, acc, profit_index, model


# -------------------------------
# Dataset Caching
# -------------------------------
def prepare_and_cache_data(df, selected_features, window_size, forecast_window, ticker):
    """Create or load cached dataset and scaler parameters."""
    cache_dir = f"cache/{ticker}_w{window_size}_f{forecast_window}"
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "y_train_raw": os.path.join(cache_dir, "y_train_raw.npy"),
        "y_val_raw": os.path.join(cache_dir, "y_val_raw.npy"),
        "scaler_X": os.path.join(cache_dir, "scaler_X_params.npy"),
        "scaler_y": os.path.join(cache_dir, "scaler_y_params.npy"),
    }

    # ---------------- Load cache if exists ----------------
    if all(os.path.exists(p) for p in paths.values()):
        print(f"✅ Loaded cached dataset for (w={window_size}, f={forecast_window})")

        # Restore scalers
        params_y = np.load(paths["scaler_y"], allow_pickle=True).item()
        scaler_y = MinMaxScaler()
        scaler_y.min_ = params_y["min"]
        scaler_y.scale_ = params_y["scale"]
        scaler_y.data_min_ = params_y["data_min"]
        scaler_y.data_max_ = params_y["data_max"]
        scaler_y.data_range_ = params_y["data_range"]

        return {**paths, "scaler_y": scaler_y}

    # ---------------- Otherwise generate dataset ----------------
    print(f"⚙️ Generating dataset for (w={window_size}, f={forecast_window})...")
    processor = DataPreprocessing(df=df)

    # Windowing + split
    X, y = processor.create_windowed_data(df[selected_features].values, window_size, forecast_window)
    X_train, X_val, _, y_train, y_val, _ = processor.split_dataset(X, y)

    # Feature scaling (fit only on train)
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_val_scaled = scaler_X.transform(X_val.reshape(X_val.shape[0], -1))
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)

    # Target scaling
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Save arrays
    np.save(paths["X_train"], X_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_train"], y_train_scaled)
    np.save(paths["y_val"], y_val_scaled)
    np.save(paths["y_train_raw"], y_train)
    np.save(paths["y_val_raw"], y_val)

    # Save scalers
    scaler_y_params = {
        "min": scaler_y.min_,
        "scale": scaler_y.scale_,
        "data_min": scaler_y.data_min_,
        "data_max": scaler_y.data_max_,
        "data_range": scaler_y.data_range_
    }
    np.save(paths["scaler_y"], scaler_y_params, allow_pickle=True)

    scaler_X_params = {
        "min": scaler_X.min_,
        "scale": scaler_X.scale_,
        "data_min": scaler_X.data_min_,
        "data_max": scaler_X.data_max_,
        "data_range": scaler_X.data_range_
    }
    np.save(paths["scaler_X"], scaler_X_params, allow_pickle=True)

    return {**paths, "scaler_y": scaler_y}


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    ticker = 'AAPL'
    os.makedirs('stock_results', exist_ok=True)

    result_file = f'stock_results/{ticker}_LR_results.csv'
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Indicators", "Window Size", "Forecast Window",
                "RMSE", "MAPE", "R2", "Accuracy", "Profit Index",
                "Intercept", "Coefficients"
            ])

    processor = DataPreprocessing(ticker=ticker)
    df_with_all_indicators = processor.add_technical_indicators()

    selected_features = ['Close', '20MA', '50MA', 'RSI', 'MACD', 'Upper_BB',
                         'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV']

    all_combinations = list(combinations(selected_features[1:], 6))
    window_forecast_combos = [(60, 1), (60, 30), (5, 1)]

    for i, indicator_combo in enumerate(all_combinations):
        for j, (w, forecast_window) in enumerate(window_forecast_combos):
            print(f"\n=== [{i+1}/{len(all_combinations)}] Combo: {indicator_combo}, (w={w}, f={forecast_window}) ===")

            cached_data = prepare_and_cache_data(
                df_with_all_indicators,
                ['Close'] + list(indicator_combo),
                w, forecast_window, ticker
            )

            X_train = np.load(cached_data["X_train"])
            X_val = np.load(cached_data["X_val"])
            y_train_scaled = np.load(cached_data["y_train"])
            y_val_scaled = np.load(cached_data["y_val"])
            scaler_y = cached_data["scaler_y"]

            rmse, mape, r2, acc, profit_index, model = train_and_evaluate_linear_model(
                X_train, y_train_scaled, X_val, y_val_scaled,
                scaler_y, forecast_window
            )

            # Log model coefficients
            intercept = float(model.intercept_)
            coefficients = ", ".join([f"{coef:.6f}" for coef in model.coef_.flatten()])

            with open(result_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ', '.join(indicator_combo),
                    w, forecast_window,
                    float(rmse), float(mape), float(r2), float(acc), float(profit_index),
                    intercept, coefficients
                ])

    record_best_models(result_file)