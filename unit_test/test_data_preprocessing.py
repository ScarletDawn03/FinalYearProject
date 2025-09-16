import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# At the top of your test file
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ReusableFunctions.DataPreprocessing import DataPreprocessing

# -------------------------
# Existing tests
# -------------------------

def test_add_technical_indicators(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    df_with_indicators = processor.add_technical_indicators()

    expected_cols = ['20MA', '50MA', 'RSI', 'MACD', 'Signal_Line',
                     'Upper_BB', 'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV']
    
    for col in expected_cols:
        assert col in df_with_indicators.columns, f"{col} missing from DataFrame"

    assert not df_with_indicators.isnull().values.any(), "DataFrame contains nulls after preprocessing"

@patch("ReusableFunctions.DataPreprocessing.yf.download")
@patch("ReusableFunctions.DataPreprocessing.yf.Ticker")
def test_download_stock_data_and_remove_dates(mock_ticker, mock_download, dummy_stock_data):
    mock_download.return_value = dummy_stock_data

    mock_ticker_instance = mock_ticker.return_value
    mock_ticker_instance.actions = pd.DataFrame({'Dividends': [0, 0, 0]}, index=pd.date_range('2020-01-01', periods=3))
    mock_ticker_instance.earnings_dates = pd.DataFrame(index=pd.date_range('2020-01-10', periods=3))

    dp = DataPreprocessing(ticker='AAPL')
    assert isinstance(dp.df, pd.DataFrame)
    expected_removed = 3
    assert len(dp.df) == len(dummy_stock_data) - expected_removed

def test_create_windowed_data(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    processor.df = processor.add_technical_indicators()
    
    selected = ['Close', '20MA', 'RSI']
    scaled = processor.df[selected]
    scaled_data = MinMaxScaler().fit_transform(scaled)

    X, y = processor.create_windowed_data(scaled_data, window_size=10, forecast_window=1)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1:] == (10, len(selected)), "Incorrect shape for windowed input"

def test_split_dataset_shapes(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    processor.df = processor.add_technical_indicators()
    
    selected = ['Close', '20MA']
    scaled = processor.df[selected]
    scaled_data = MinMaxScaler().fit_transform(scaled)

    X, y = processor.create_windowed_data(scaled_data, window_size=10, forecast_window=1)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(X, y)

    total = len(X)
    assert len(X_train) + len(X_val) + len(X_test) == total
    assert len(y_train) == len(X_train)

def test_indicator_value_ranges(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    df_indicators = processor.add_technical_indicators()

    assert df_indicators['RSI'].between(0, 100).all(), "RSI values out of range [0,100]"
    assert (df_indicators['ATR'] >= 0).all(), "ATR has negative values"
    # OBV is cumulative, should be numeric
    assert np.issubdtype(df_indicators['OBV'].dtype, np.number), "OBV is not numeric"

def test_forecast_window_greater_than_one(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    processor.df = processor.add_technical_indicators()

    selected = ['Close', '20MA', 'RSI']
    scaled_data = MinMaxScaler().fit_transform(processor.df[selected])

    X_fw1, y_fw1 = processor.create_windowed_data(scaled_data, window_size=10, forecast_window=1)
    X_fw5, y_fw5 = processor.create_windowed_data(scaled_data, window_size=10, forecast_window=5)

    assert X_fw5.shape[0] == y_fw5.shape[0]
    assert X_fw5.shape[1:] == (10, len(selected)), "Incorrect X shape for forecast_window > 1"
    # y_fw5 should be shifted compared to y_fw1
    assert not np.array_equal(y_fw1[:len(y_fw5)], y_fw5), "y values for different forecast windows should differ"

def test_prepare_data_returns_scaled_sets(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    processor.df = processor.add_technical_indicators()

    selected = ['Close', '20MA', 'RSI']
    result = processor.prepare_data(selected, window_size=10, forecast_window=1)

    X_train, X_val = result["X_train"], result["X_val"]
    y_train, y_val = result["y_train"], result["y_val"]

    # Shapes match
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]

    # Training data should be in [0,1] within tolerance
    assert np.all((X_train >= -1e-8) & (X_train <= 1 + 1e-8)), "X_train not scaled properly"
    assert np.all((y_train >= -1e-8) & (y_train <= 1 + 1e-8)), "y_train not scaled properly"


