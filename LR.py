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
    """Train Linear Regression and evaluate on train & validation sets."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_flat, y_train)

    # 🔹 Training predictions
    train_preds_scaled = model.predict(X_train_flat)
    train_preds_unscaled = scaler_y.inverse_transform(train_preds_scaled.reshape(-1, 1))
    y_train_true_unscaled = scaler_y.inverse_transform(y_train.reshape(-1, 1))

    # 🔹 Validation predictions
    val_preds_scaled = model.predict(X_val_flat)
    val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1))
    y_val_true_unscaled = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    # Training metrics
    train_rmse = EM.rmse(y_train_true_unscaled, train_preds_unscaled)
    train_mape = EM.mape(y_train_true_unscaled, train_preds_unscaled)
    train_r2 = EM.r2(y_train_true_unscaled, train_preds_unscaled)
    train_acc = EM.accuracy(y_train_true_unscaled, train_preds_unscaled)
    train_profit_index = EM.profitability_index(y_train_true_unscaled, train_preds_unscaled, forecast_window)

    # Validation metrics
    val_rmse = EM.rmse(y_val_true_unscaled, val_preds_unscaled)
    val_mape = EM.mape(y_val_true_unscaled, val_preds_unscaled)
    val_r2 = EM.r2(y_val_true_unscaled, val_preds_unscaled)
    val_acc = EM.accuracy(y_val_true_unscaled, val_preds_unscaled)
    val_profit_index = EM.profitability_index(y_val_true_unscaled, val_preds_unscaled, forecast_window)

    # 🔹 Console log: Train vs Val
    print("\n--- Performance ---")
    print(f"[Train] RMSE={train_rmse:.4f}, MAPE={train_mape:.4f}, R²={train_r2:.4f}, "
          f"Accuracy={train_acc:.4f}, Profit Index={train_profit_index:.4f}")
    print(f"[Val]   RMSE={val_rmse:.4f}, MAPE={val_mape:.4f}, R²={val_r2:.4f}, "
          f"Accuracy={val_acc:.4f}, Profit Index={val_profit_index:.4f}")

    # 🔹 Only return validation metrics for CSV writing
    return val_rmse, val_mape, val_r2, val_acc, val_profit_index



# -------------------------------
# Dataset Preparation (No Cache)
# -------------------------------
def prepare_data(df, selected_features, window_size, forecast_window):
    """Always generate dataset fresh (no caching)."""
    print(f"⚙️ Generating dataset for (w={window_size}, f={forecast_window})...")
    processor = DataPreprocessing(df=df)

    # Windowing + split
    X, y = processor.create_windowed_data(
        df[selected_features].values,
        window_size,
        forecast_window
    )
    X_train, X_val, _, y_train, y_val, _ = processor.split_dataset(X, y)

    # Feature scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_val_scaled = scaler_X.transform(X_val.reshape(X_val.shape[0], -1))
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)

    # Target scaling
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "y_train_raw": y_train,
        "y_val_raw": y_val,
        "scaler_y": scaler_y
    }


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    ticker = 'BK'
    os.makedirs('stock_results', exist_ok=True)

    result_file = f'stock_results/{ticker}_LR_results.csv'
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
    window_forecast_combos = [(60, 1), (60, 30), (5, 1)]

    for i, indicator_combo in enumerate(all_combinations):
        for j, (w, forecast_window) in enumerate(window_forecast_combos):
            print(f"\n=== [{i+1}/{len(all_combinations)}] Combo: {indicator_combo}, (w={w}, f={forecast_window}) ===")

            data = prepare_data(
                df_with_all_indicators,
                ['Close'] + list(indicator_combo),
                w, forecast_window
            )

            X_train = data["X_train"]
            X_val = data["X_val"]
            y_train_scaled = data["y_train"]
            y_val_scaled = data["y_val"]
            scaler_y = data["scaler_y"]

            rmse, mape, r2, acc, profit_index = train_and_evaluate_linear_model(
                X_train, y_train_scaled, X_val, y_val_scaled,
                scaler_y, forecast_window
            )

            with open(result_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ', '.join(indicator_combo),
                    w, forecast_window,
                    float(rmse), float(mape), float(r2), float(acc), float(profit_index)
                ])

    record_best_models(result_file)
