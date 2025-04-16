import pandas as pd
import numpy as np
import yfinance as yf

class TechnicalIndicators:
    def __init__(self, ticker=None, df=None, start_date='2010-01-01', end_date='2024-12-31'):
        """
        Initialize the TechnicalIndicators class with either a DataFrame or a ticker symbol.
        
        Args:
        - ticker (str, optional): Ticker symbol for stock data (if df is not provided).
        - df (pd.DataFrame, optional): Pre-loaded stock data (if provided, ticker is ignored).
        - start_date (str): Start date for downloading stock data (if ticker is used).
        - end_date (str): End date for downloading stock data (if ticker is used).
        """
        if df is not None:
            self.df = df
        elif ticker is not None:
            self.df = self.download_stock_data(ticker, start_date, end_date)
        else:
            raise ValueError("Either a ticker or a DataFrame must be provided.")

    def download_stock_data(self, ticker, start_date='2010-01-01', end_date='2024-12-31'):
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

    def add_technical_indicators(self):
        """
        Adds common technical indicators to the stock data DataFrame.
        
        Returns:
        - pd.DataFrame: The original DataFrame with added technical indicators.
        """
        pd.options.mode.chained_assignment = None  # Disable warnings for chained assignments

        # Moving Averages
        self.df['20MA'] = self.df['Close'].rolling(window=20, min_periods=1).mean()
        self.df['50MA'] = self.df['Close'].rolling(window=50, min_periods=1).mean()
        self.df['200MA'] = self.df['Close'].rolling(window=200, min_periods=1).mean()

        # Relative Strength Index (RSI)
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'].fillna(50, inplace=True)

        # MACD and Signal Line
        self.df['12EMA'] = self.df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        self.df['26EMA'] = self.df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        self.df['MACD'] = self.df['12EMA'] - self.df['26EMA']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

        # Bollinger Bands
        self.df['20STD'] = self.df['Close'].rolling(window=20, min_periods=1).std()
        self.df['Upper_BB'] = self.df['20MA'] + (self.df['20STD'] * 2)
        self.df['Lower_BB'] = self.df['20MA'] - (self.df['20STD'] * 2)

        # Commodity Channel Index (CCI)
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        mean_dev = lambda x: np.mean(np.abs(x - np.mean(x)))
        self.df['CCI'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / \
                        (0.015 * typical_price.rolling(window=20, min_periods=1).apply(mean_dev, raw=True))

        # Average True Range (ATR)
        self.df['TR'] = np.maximum(self.df['High'] - self.df['Low'], 
                                   np.maximum(abs(self.df['High'] - self.df['Close'].shift(1)), 
                                              abs(self.df['Low'] - self.df['Close'].shift(1))))
        self.df['ATR'] = self.df['TR'].rolling(window=14, min_periods=1).mean()
        self.df.drop(columns=['TR'], inplace=True)  # Drop intermediate column

        # Rate of Change (ROC)
        self.df['ROC'] = self.df['Close'].pct_change(periods=10) * 100

        # Williams %R
        self.df['Williams_%R'] = ((self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Close']) / 
                                  (self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Low'].rolling(window=14, min_periods=1).min())) * -100

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        # Drop intermediate columns
        self.df.drop(columns=['20STD'], inplace=True)

        # Forward-fill and backward-fill NaN values
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Verify no NaN values exist
        print("Null values in each column:\n", self.df.isnull().sum())
        print(f"Does the dataset contain any null values? {self.df.isnull().values.any()}")

        return self.df  # Return the DataFrame with technical indicators added
