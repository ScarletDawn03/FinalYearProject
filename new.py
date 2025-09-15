from ReusableFunctions.RecordBestModel import record_best_models

# Set the model you want to process
model = "LR"  # Can be "LSTM", "CNN", or "LR"

# List of tickers
tickers = ["0097.KL", "0166.KL", "1023.KL", "5258.KL", "AAPL", "QCOM", "C", "BK"]

# Execute record_best_models for all tickers with the selected model
for ticker in tickers:
    file_path = f"stock_results/{ticker}_{model}_results.csv"
    print(f"Processing {file_path} with model = {model}")
    record_best_models(file_path)
