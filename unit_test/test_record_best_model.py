import pandas as pd
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ReusableFunctions.RecordBestModel import record_best_models
@pytest.fixture
def dummy_model_csv(tmp_path):
    # Create a dummy CSV file
    data = {
        "Forecast Window": [1,1,1,1,1,30,30,30,30,30],
        "Accuracy": [0.8,0.85,0.9,0.7,0.6,0.9,0.92,0.88,0.85,0.8],
        "Profit Index": [1.1,1.3,1.2,1.0,0.9,1.5,1.4,1.6,1.2,1.1],
        "R2": [0.5,0.6,0.7,0.8,0.9,0.95,0.85,0.8,0.75,0.7],
        "Other_Col": list(range(10))
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "AAPL_LSTM_results.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_record_best_models_creates_file(tmp_path, dummy_model_csv):
    output_dir = tmp_path / "best_results"
    record_best_models(str(dummy_model_csv), output_dir=str(output_dir))

    # Check if output file exists
    output_files = os.listdir(output_dir)
    assert len(output_files) == 1
    output_file = output_dir / output_files[0]

    # Check CSV content
    df_out = pd.read_csv(output_file)
    assert df_out["Forecast Window"].isin([1,30]).all()
    assert df_out.shape[0] <= 10  # top 5 per forecast window
    assert list(df_out.columns) == ["Forecast Window", "Accuracy", "Profit Index", "R2", "Other_Col"]

def test_record_best_models_excludes_negative_r2(tmp_path):
    # CSV with negative R2
    data = {
        "Forecast Window": [1,1,1],
        "Accuracy": [0.8,0.9,0.7],
        "Profit Index": [1.0,1.1,1.2],
        "R2": [-0.1,-0.5,0.3]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "TEST_LR_results.csv"
    df.to_csv(file_path, index=False)

    output_dir = tmp_path / "best_results"
    record_best_models(str(file_path), output_dir=str(output_dir))

    df_out = pd.read_csv(output_dir / "TEST_LR_best.csv")
    assert df_out["R2"].min() >= 0
