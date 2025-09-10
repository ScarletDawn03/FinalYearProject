import pandas as pd
import os

def record_best_models(input_csv_path, output_dir="best_results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv_path)

    # Get original column order to preserve header structure
    columns = df.columns.tolist()

    best_models = []

    for forecast_horizon in [1, 30]:
        df_fh = df[df["Forecast Window"] == forecast_horizon]
        if df_fh.empty:
            continue

        # Find max accuracy
        max_acc = df_fh["Accuracy"].max()
        top_acc_df = df_fh[df_fh["Accuracy"] == max_acc]

        # Break tie using profitability index
        if len(top_acc_df) > 1:
            best_row = top_acc_df.loc[top_acc_df["Profit Index"].idxmax()]
        else:
            best_row = top_acc_df.iloc[0]

        best_models.append(best_row)

    if best_models:
        result_df = pd.DataFrame(best_models)

        # Ensure consistent column ordering
        result_df = result_df[columns]

        ticker_symbol = os.path.basename(input_csv_path).split("_")[0]
        output_path = os.path.join(output_dir, f"{ticker_symbol}_LR_best.csv")
        result_df.to_csv(output_path, index=False)
        print(f"✅ Best models saved to: {output_path}")
    else:
        print("⚠️ No best models found (check if forecast horizon 1/30 exists in the file).")
