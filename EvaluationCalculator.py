import pandas as pd
import numpy as np

# Path to the results CSV file
csv_path = "stock_results/AAPL_results_CNN.csv"

# Read the CSV file into a pandas DataFrame
df_results = pd.read_csv(csv_path)

# Extract relevant columns starting from the RMSE column
# Assuming the columns are in the following order after the hyperparameters:
# RMSE, MAPE, R2, Accuracy, Train Loss, Validation Loss

metrics_columns = ["RMSE", "MAPE", "R2", "Accuracy"]

# Calculate the statistics for each metric
statistics = {}
for column in metrics_columns:
    statistics[column] = {
        "Min": df_results[column].min(),
        "Max": df_results[column].max(),
        "Average": df_results[column].mean(),
        "Std Dev": df_results[column].std()
    }

# Display the statistics
for metric, stats in statistics.items():
    print(f"Statistics for {metric}:")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value:.4f}")
    print("\n")
