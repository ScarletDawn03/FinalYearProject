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



📂 Project Structure
├── best_results/                   # Top 5 configurations of models for the different combinations of forecast horizons are stored here
├── check/                          # Raw stock datasets & Technical Indicators feature set for manual checking. 
├── ReusableFunctions/             # Contains reusable classes of functions that are utilized in the models.
│   ├── DataPreprocessing.py        # LSTM, CNN, LR reusable functions (Preprocessing steps) are stored here.
│   ├── EvaluationMetrics.py        # Helper functions (metrics).
│   └── RecordBestModel.py          # Script to record top 5 configuration of model for 1 and 30 day forecast horizon; executed after each run of model
├── stock_results/                  # Saved outputs of LSTM, CNN and LR models for hyperparameter tuning (LSTM&CNN should have 15751 lines; LR should have 631); Utilizes Training and Validation ONLY
├── TestResults/                    # Consist of Test Results of Dataset Utilized in this research using ipynb file for utilize graph visualization feature. Might require extension in VS.Code.  Utilizes Training, Validation and Test.
├── unit_test/                      # Consists of automated testing test suites for all ReusableFunctions.
├── CNN.py                          # CNN script for hyperparameter tuning; Utilizes Training and Validation ONLY; Output in stock_results based on ticker
├── CNNFIX.py                       # Due to CNN model requiring long hours to execute, PC may overheat, freeze or crash. This script can be used for continuation from where the previous CNN run was cut off. Details on how to use are documented within code.
├── LR.py                           # Consists of automated testing test suites for all ReusableFunctions.
├── LSTM.py                         # LSTM script for hyperparameter tuning; Utilizes Training and Validation ONLY; Output in stock_results based on ticker
├── LSTMFIX.py                      # Due to LSTM model requiring long hours to execute, PC may overheat, freeze or crash. This script can be used for continuation from where the previous LSTM run was cut off. Details on how to use are documented within code.
├── new.py                          # A script that regenerates the best configurations of respective models based on results in stock_results
├── reproducibility_settings.py     # Contains seed which ensures the weights and variability due to randomness remains consistent
└── README.md                       # Project documentation

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


📊 Results Summary

LSTM: Best at capturing temporal dependencies and achieving high accuracy.

CNN: Strong performance in detecting local patterns and momentum indicators.

LR: Lightweight baseline with competitive performance in less volatile stocks.


📌 Research Contribution

This work:

Benchmarks three widely used models in stock market prediction.

Provides comparative analysis across different markets and sectors.

Investigates technical indicators, window sizes, forecast horizons, and hyperparameters.

Suggests practical insights for intra-day to medium-term trading strategies.

DEMONSTRATION HOW TO SCRIPT WORKS IS UPLOADED IN THIS TUTORIAL: https://youtu.be/rQ_TAfjQ5tg


