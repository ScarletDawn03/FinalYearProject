import pandas as pd
import os
import tempfile
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReusableFunctions")))

from RecordBestModel import record_best_models

def test_record_best_models(tmp_path):
    # Simulate input CSV
    data = {
        "Forecast Window": [1, 1, 30, 30],
        "Accuracy": [0.8, 0.85, 0.78, 0.78],
        "Profit Index": [1.2, 1.1, 2.0, 1.5],
        "OtherCol": [123, 456, 789, 101],
    }
    df = pd.DataFrame(data)
    input_path = tmp_path / "AAPL_LSTM_results.csv"
    df.to_csv(input_path, index=False)

    # Call the function
    record_best_models(str(input_path), output_dir=str(tmp_path))

    # Check if output file exists
    output_file = tmp_path / "AAPL_LSTM_best.csv"
    assert output_file.exists()

    # Check contents
    result = pd.read_csv(output_file)
    assert len(result) == 2  # One for forecast=1 and one for forecast=30
    assert result.iloc[0]["Accuracy"] == 0.85  # Best accuracy for forecast=1
    assert result.iloc[1]["Profit Index"] == 2.0  # Best profit index for forecast=30 (tie)

