A Study of Stock Market Prediction using Various ML & DL Technique with Various Datasets and Hyperparameters

This repository contains the implementation and experimental analysis of Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and Linear Regression (LR) models for stock market prediction. The project investigates the predictive performance of these models across different datasets, sectors, and trading volumes, with evaluation based on RMSE, MAE, R², Accuracy (within ±5% threshold), and Profitability Index as part of my Final Year Project.

📖 Project Overview

Focuses on technical analysis using OHLCV data and technical indicators (e.g., MA, RSI, MACD).

Implements three widely used models in structured literature review: LSTM, CNN, and LR.

Evaluates performance across NYSE, NASDAQ, and Bursa Malaysia datasets.

Provides insights into window size, forecast horizon, and hyperparameter tuning for improved predictive accuracy.

⚙️ Environment Setup
1. Clone Repository
cd <repo name>

2. Create Virtual Environment

It’s recommended to use either a docker file (Pytorch) or venv.

Using docker file: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html

Download from ROCm official site within directory:

Or using venv:

python3 -m venv venv
source venv/bin/activate   # for Linux


3. Install Dependencies where necessary
pip install -r requirements.txt

4. Additionally if you would like to use GPU acceleration, you can opt to set up CUDA (nvidia) or ROCm (AMD), at which the latter option is used in this research (ROCM6.3.2).



## 📂 Project Structure

```text
FinalYearProject/
├── best_results/                   # 24 CSVs: selected configurations for each ticker/model
│   └── <ticker>_<model>_best.csv
├── stock_results/                  # 24 CSVs: full hyperparameter tuning results
│   └── <ticker>_<model>_results.csv
├── TestResults/                    # Test evaluation notebooks with saved metrics and plots
│   ├── CNN/                        # 8 ticker-specific CNN notebooks
│   ├── LR/                         # 8 ticker-specific linear regression notebooks
│   ├── LSTM/                       # 8 ticker-specific LSTM notebooks
│   ├── CNNTemplate.ipynb
│   ├── LRTemplate.ipynb
│   └── LSTMTemplate.ipynb
├── ReusableFunctions/
│   ├── DataPreprocessing.py         # Stock data download, cleaning and technical indicators
│   ├── EvaluationMetrics.py         # Evaluation metric helpers
│   └── RecordBestModel.py           # Selects up to 5 configurations per forecast horizon
├── unit_test/
│   ├── conftest.py
│   ├── test_data_preprocessing.py
│   ├── test_evaluation_metrics.py
│   └── test_record_best_model.py
├── CNN.py                          # CNN hyperparameter tuning using training/validation data
├── CNNFIX.py                       # Resume interrupted CNN tuning runs; see instructions in code
├── LR.py                           # Linear regression tuning using training/validation data
├── LSTM.py                         # LSTM hyperparameter tuning using training/validation data
├── LSTMFIX.py                      # Resume interrupted LSTM tuning runs; see instructions in code
├── new.py                          # Regenerate best configurations from stock_results/
├── reproducibility_settings.py     # Shared random seed settings
├── requirements.txt                # Python dependencies
├── packages.microsoft.gpg          # Microsoft package signing key file
├── .gitignore
└── README.md
```

The tuning scripts also create `check/<ticker>_<model>_downloaded_data.csv` for manual inspection of downloaded data and technical indicators. The `check/` directory is ignored by Git and is not included in this checkout.

NOTE: Please delete specific file in *stock_results* given that a hyperparameter tuning script is to be re-run on an existing ticker as it will append onto existing results.

🚀 Running Experiments
Train a Model
Current Directory: python3 {scipt name}

Available Models

LSTM

CNN

LR

Evaluation

The models are evaluated using:

RMSE, MAE, R², Accuracy (prediction within ±5% of actual closing price), Profitability Index


## 📊 Where to Find the Results

**Results are already included in this repository.** You can inspect the CSV files and saved notebook outputs without rerunning the experiments.

| Result type | Location | Contents |
| --- | --- | --- |
| Full hyperparameter tuning results | [stock_results/](stock_results/) | Training/validation results for each ticker and model, including configurations, R², Accuracy and Profit Index. These CSVs are not the final test-set evaluation. |
| Best configurations | [best_results/](best_results/) | Up to five configurations for each of the 1-day and 30-day forecast horizons, filtered to nonnegative R² and ranked by Accuracy, then Profit Index (both descending). |
| Test-set evaluation | [TestResults/CNN/](TestResults/CNN/), [TestResults/LSTM/](TestResults/LSTM/), [TestResults/LR/](TestResults/LR/) | Ticker-specific notebooks containing saved evaluation outputs and plots from experiments using training, validation and test data. Open them in Jupyter or VS Code with notebook support to inspect the cell outputs. |

Both CSV directories contain 24 files covering three models (`CNN`, `LSTM`, `LR`) and eight tickers: `0097.KL`, `0166.KL`, `1023.KL`, `5258.KL`, `AAPL`, `BK`, `C` and `QCOM`. Each model's test notebook directory contains eight ticker-specific notebooks. The three template notebooks at the root of `TestResults/` provide starting points for further evaluations.

For example, to inspect Apple's CNN experiment:

- Full tuning results: [stock_results/AAPL_CNN_results.csv](stock_results/AAPL_CNN_results.csv).
- Selected configurations: [best_results/AAPL_CNN_best.csv](best_results/AAPL_CNN_best.csv).
- Test metrics and visualizations: [TestResults/CNN/AAPL_CNN.ipynb](TestResults/CNN/AAPL_CNN.ipynb).

CSV filenames follow `<ticker>_<model>_results.csv` and `<ticker>_<model>_best.csv`. Test notebooks generally follow `<ticker>_<model>.ipynb`; the LSTM notebook for `0097.KL` is named [0097_LSTM.ipynb](TestResults/LSTM/0097_LSTM.ipynb).

Running `CNN.py`, `LSTM.py` or `LR.py` writes tuning results to `stock_results/`. To regenerate the selected configurations from existing CSVs, run `python3 new.py` from the repository root; its current model selection is `LSTM`. See `ReusableFunctions/RecordBestModel.py` for the selection rules.

📌 Research Contribution

This work:

Benchmarks three widely used models in stock market prediction.

Provides comparative analysis across different markets and sectors.

Investigates technical indicators, window sizes, forecast horizons, and hyperparameters.

Suggests practical insights for intra-day to medium-term trading strategies.

DEMONSTRATION HOW TO SCRIPT WORKS IS UPLOADED IN THIS TUTORIAL: https://youtu.be/rQ_TAfjQ5tg


