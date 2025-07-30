import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing:
    def __init__(self, ticker=None, df=None, start_date='2014-01-01', end_date='2024-12-31'):
        """
        Initialize the TechnicalIndicators class with either a DataFrame or a ticker symbol.
        
        Args:
        - ticker (str, optional): Ticker symbol for stock data (if df is not provided).
        - df (pd.DataFrame, optional): Pre-loaded stock data (if provided, ticker is ignored).
        - start_date (str): Start date for downloading stock data (if ticker is used).
        - end_date (str): End date for downloading stock data (if ticker is used).
        """

        self.analysis_start_date = '2015-01-01'
        if df is not None:
            self.df = df
        elif ticker is not None:
            self.df = self.download_stock_data(ticker, start_date, end_date)
            self.remove_exdividend_and_earnings_dates(ticker)
        else:
            raise ValueError("Either a ticker or a DataFrame must be provided.")

    def download_stock_data(self, ticker, start_date='2014-01-01', end_date='2024-12-31'):
        """
        Downloads stock data using yfinance for the given ticker and date range.
        
        Args:
        - ticker (str): The stock ticker symbol.
        - start_date (str): The start date for the stock data.
        - end_date (str): The end date for the stock data.
        
        Returns:
        - pd.DataFrame: Stock data for the specified ticker and date range.
        """
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    
    def remove_exdividend_and_earnings_dates(self, ticker):
        ticker_obj = yf.Ticker(ticker)

        # Get ex-dividend dates
        try:
            ex_dividends = ticker_obj.actions[ticker_obj.actions['Dividends'] > 0].index
            ex_dividends = ex_dividends.tz_localize(None).normalize()
        except Exception as e:
            print("Could not retrieve ex-dividend dates:", e)
            ex_dividends = pd.Index([])

        # Get earnings dates
        try:
            earnings_calendar = ticker_obj.earnings_dates
            earnings_dates = earnings_calendar.index
            earnings_dates = earnings_dates.tz_localize(None).normalize()
        except Exception as e:
            print("Could not retrieve earnings dates:", e)
            earnings_dates = pd.Index([])

        # Combine all dates
        combined_dates = ex_dividends.append(earnings_dates).drop_duplicates()

        # Normalize your dataframe index
        self.df.index = self.df.index.tz_localize(None).normalize()

        # Remove from DataFrame
        before = len(self.df)
        self.df = self.df[~self.df.index.isin(combined_dates)]
        after = len(self.df)

        print(f"Removed {before - after} rows corresponding to ex-dividend and earnings dates.")

    def add_technical_indicators(self):
        """
        Adds common technical indicators to the stock data DataFrame.
        
        Returns:
        - pd.DataFrame: The original DataFrame with added technical indicators.
        """
        pd.options.mode.chained_assignment = None  # Disable warnings for chained assignments

        # Forward-fill and backward-fill NaN values
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Moving Averages
        self.df['20MA'] = self.df['Close'].rolling(window=20, min_periods=1).mean() #Takes sum of closing price of the last 20 days and divide by 20
        self.df['50MA'] = self.df['Close'].rolling(window=50, min_periods=1).mean() #Takes sum of closing price of the last 50 days and divide by 50

        # Relative Strength Index (RSI)
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean() #Sum of positive values over 14 days
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean() #Sum of negative values over 14 days
        rs = gain / loss #Calculate the relative strength
        self.df['RSI'] = 100 - (100 / (1 + rs)) #>70 overbought, <30 oversold
        self.df['RSI'].fillna(50, inplace=True)

        # MACD and Signal Line
        self.df['12EMA'] = self.df['Close'].ewm(span=12, adjust=False, min_periods=1).mean() #EMA=Soothing Factor x Price + (1-Soothing Factor) x Previous EMA
        self.df['26EMA'] = self.df['Close'].ewm(span=26, adjust=False, min_periods=1).mean() #Soothing Factor= 2/(span)+1
        self.df['MACD'] = self.df['12EMA'] - self.df['26EMA']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

        # Bollinger Bands
        self.df['20STD'] = self.df['Close'].rolling(window=20, min_periods=1).std() #Standard deviation of Close over 20 periods
        self.df['Upper_BB'] = self.df['20MA'] + (self.df['20STD'] * 2)  # 20MA + 2 x Standard Deviation over a 20 week period
        self.df['Lower_BB'] = self.df['20MA'] - (self.df['20STD'] * 2)  # 20MA - 2 x Standard Deviation over a 20 week period

        # Commodity Channel Index (CCI)
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        mean_dev = lambda x: np.mean(np.abs(x - np.mean(x))) #sum of absolute value of previous typical price- typical price over a span of 20 weeks divide by the number of days
        self.df['CCI'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / \
                        (0.015 * typical_price.rolling(window=20, min_periods=1).apply(mean_dev, raw=True)) #Typical Price -Typical Price Over a Span of 20 wewks/0.015 x Mean Diviation

        # Average True Range (ATR)
        self.df['TR'] = np.maximum(self.df['High'] - self.df['Low'], 
                                   np.maximum(abs(self.df['High'] - self.df['Close'].shift(1)), 
                                              abs(self.df['Low'] - self.df['Close'].shift(1))))
        self.df['ATR'] = self.df['TR'].rolling(window=14, min_periods=1).mean() #Simple Moving Average of TR Over a 14 Week Period
        self.df.drop(columns=['TR'], inplace=True)  # Drop intermediate column

        # Williams %R
        self.df['Williams_%R'] = ((self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Close']) / 
                                  (self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Low'].rolling(window=14, min_periods=1).min())) * -100   # (Highest High in the past 14 days - Close)/ (Highest High in the past 14 days-Lowest Low in the past 14 days) x -100

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum() #Close > yesterday's, add todays volume to OBV ; Close < yesterday's, subtract todays volume from OBV

        # Drop intermediate columns
        self.df.drop(columns=['20STD'], inplace=True)

        # Verify no NaN values exist
        print("Null values in each column:\n", self.df.isnull().sum())
        print(f"Does the dataset contain any null values? {self.df.isnull().values.any()}")

        
        # Return only rows from analysis start date forward
        return self.df.loc[self.analysis_start_date:]
    
    def normalize_indicator_combinations(self, all_indicators: list[str]):
        indicator_combinations = list(combinations(all_indicators, 5)) 
        all_scaled_data = {}
        for selected_indicators in indicator_combinations:
            selected_features = ['Close'] + list(selected_indicators)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.df[selected_features])
            all_scaled_data[selected_indicators] = (scaled_data, scaler)
        return all_scaled_data

    def create_windowed_data(self, scaled_data, window_size: int = 50, forecast_window: int = 1):
        X, y = [], []
        for i in range(window_size, len(scaled_data) - forecast_window + 1):
            X.append(scaled_data[i - window_size:i])
            y.append(scaled_data[i + forecast_window - 1][0])  # assuming Close is the first feature
        return np.array(X), np.array(y)

    def split_dataset(self, X: np.ndarray, y: np.ndarray, train_size: float = 0.7, val_size: float = 0.1):
        total_samples = len(X)
        train_end = int(total_samples * train_size)
        val_end = train_end + int(total_samples * val_size)
        
        return (
            X[:train_end], X[train_end:val_end], X[val_end:],
            y[:train_end], y[train_end:val_end], y[val_end:]
        )
    
    def split_train_test_80_20(self, X: np.ndarray, y: np.ndarray):
        """
        Splits the dataset into 80% training and 20% testing sets.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        total_samples = len(X)
        split_index = int(total_samples * 0.8)
        
        return (
            X[:split_index], X[split_index:],
            y[:split_index], y[split_index:]
        )
