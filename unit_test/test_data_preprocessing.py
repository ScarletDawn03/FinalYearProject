import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.preprocessing import MinMaxScaler

# At the top of your test file
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReusableFunctions")))

from DataPreprocessing import DataPreprocessing

# Test1: test_add_technical_indicators
# Uses mock data form the dummy_stock_data fixture.
# Calls add_technical_indicators() on it.
# Asserts that expected indicator colums exist
# Ensure no NaNs are left in the dataframe
def test_add_technical_indicators(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    df_with_indicators = processor.add_technical_indicators()

    # Assert technical indicators exist
    expected_cols = ['20MA', '50MA', 'RSI', 'MACD', 'Signal_Line',
                     'Upper_BB', 'Lower_BB', 'CCI', 'ATR', 'Williams_%R', 'OBV']
    
    for col in expected_cols:
        assert col in df_with_indicators.columns, f"{col} missing from DataFrame"

    assert not df_with_indicators.isnull().values.any(), "DataFrame contains nulls after preprocessing"

# @patch decorates moch yfinance.download() and yfinance.Ticker()
# mock_download.return_value provides fake dummy_stock_data instead of downloading real stock data
# Stimulates dividend and earnings date
# Initializes a DataPreprocessing(ticker='AAPL'), which internally calls yf.download() and yf.Ticker()- Both return mock data
#Assertion that some rows were removed after excluding earnings and dividend date
@patch("DataPreprocessing.yf.download")
@patch("DataPreprocessing.yf.Ticker")
def test_download_stock_data_and_remove_dates(mock_ticker, mock_download, dummy_stock_data):
    # Mock download
    mock_download.return_value = dummy_stock_data

    # Mock actions (ex-dividend) and earnings
    mock_ticker_instance = mock_ticker.return_value
    mock_ticker_instance.actions = pd.DataFrame({'Dividends': [0, 0, 0]}, index=pd.date_range('2020-01-01', periods=3))
    mock_ticker_instance.earnings_dates = pd.DataFrame(index=pd.date_range('2020-01-10', periods=3))

    dp = DataPreprocessing(ticker='AAPL')
    assert isinstance(dp.df, pd.DataFrame)
    expected_removed = 3  # or however many dates intersect
    assert len(dp.df) == len(dummy_stock_data) - expected_removed


# Uses mock data with add_technical_indicators() applied
# Selects 3 columns: 'Close', '20MA', RSI'
# Applies MinMaxScaler to scale data
# Cenerates windowed input (X) and labels (y)
#Asserts the shape of X and y matcbes expected dimensions
# Verify that time series windows are formed correctly for model traning
def test_create_windowed_data(dummy_stock_data):
    processor = DataPreprocessing(df=dummy_stock_data)
    processor.df = processor.add_technical_indicators()
    
    selected = ['Close', '20MA', 'RSI']
    scaled = processor.df[selected]
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(scaled)

    X, y = processor.create_windowed_data(scaled_data, window_size=10, forecast_window=1)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1:] == (10, len(selected)), "Incorrect shape for windowed input"

# Adds indicators, select features, scales them
# Calls create_windowed_data() to get X and y.
# Splits into train, val, test via split_dataset(.....)
# To confirm dataset splitting logic works and samples align correctly
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
