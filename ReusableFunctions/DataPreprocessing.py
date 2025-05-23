import pandas as pd
import numpy as np
import yfinance as yf

class DataPreprocessing:
    def __init__(self, ticker=None, df=None, start_date='2010-01-01', end_date='2024-12-31'):
        """
        Initialize the TechnicalIndicators class with either a DataFrame or a ticker symbol.
        
        Args:
        - ticker (str, optional): Ticker symbol for stock data (if df is not provided).
        - df (pd.DataFrame, optional): Pre-loaded stock data (if provided, ticker is ignored).
        - start_date (str): Start date for downloading stock data (if ticker is used).
        - end_date (str): End date for downloading stock data (if ticker is used).
        """

        self.analysis_start_date = '2011-01-01'
        if df is not None:
            self.df = df
        elif ticker is not None:
            self.df = self.download_stock_data(ticker, start_date, end_date)
            self.remove_exdividend_and_earnings_dates(ticker)
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
    
    def remove_exdividend_and_earnings_dates(self, ticker):
        """
        Removes rows from the DataFrame that correspond to ex-dividend and earnings dates.

        Args:
        - ticker (str): Ticker symbol to fetch the actions and earnings dates from.
        """
        ticker_obj = yf.Ticker(ticker)

        # Get ex-dividend dates
        try:
            ex_dividends = ticker_obj.actions[ticker_obj.actions['Dividends'] > 0].index
        except Exception as e:
            print("Could not retrieve ex-dividend dates:", e)
            ex_dividends = []

        # Get earnings dates
        try:
            earnings_calendar = ticker_obj.earnings_dates
            earnings_dates = earnings_calendar.index
        except Exception as e:
            print("Could not retrieve earnings dates:", e)
            earnings_dates = []

        # Combine all dates to exclude
        combined_dates = ex_dividends.append(earnings_dates)
        dates_to_exclude = combined_dates.drop_duplicates()


        # Remove from DataFrame
        before = len(self.df)
        self.df = self.df[~self.df.index.isin(dates_to_exclude)]
        after = len(self.df)

        print(f"Removed {before - after} rows corresponding to ex-dividend and earnings dates.")

    def add_technical_indicators(self):
        """
        Adds common technical indicators to the stock data DataFrame.
        
        Returns:
        - pd.DataFrame: The original DataFrame with added technical indicators.
        """
        pd.options.mode.chained_assignment = None  # Disable warnings for chained assignments

        # Moving Averages
        self.df['20MA'] = self.df['Close'].rolling(window=20, min_periods=1).mean() #Takes sum of closing price of the last 20 days and divide by 20
        self.df['50MA'] = self.df['Close'].rolling(window=50, min_periods=1).mean() #Takes sum of closing price of the last 50 days and divide by 50
        self.df['200MA'] = self.df['Close'].rolling(window=200, min_periods=1).mean() #Takes closing price of the last 200 days and divide by 200

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

        # Rate of Change (ROC)
        self.df['ROC'] = self.df['Close'].pct_change(periods=10) * 100 # (Closing Price- Closing price 10 weeks ago)/Closing price 10 days ago X 100, ROC>0 =Positive Momentum, ROC<0 = Negative Momentum

        # Williams %R
        self.df['Williams_%R'] = ((self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Close']) / 
                                  (self.df['High'].rolling(window=14, min_periods=1).max() - self.df['Low'].rolling(window=14, min_periods=1).min())) * -100   # (Highest High in the past 14 days - Close)/ (Highest High in the past 14 days-Lowest Low in the past 14 days) x -100

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum() #Close > yesterday's, add todays volume to OBV ; Close < yesterday's, subtract todays volume from OBV

        # Drop intermediate columns
        self.df.drop(columns=['20STD'], inplace=True)

        # Forward-fill and backward-fill NaN values
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Verify no NaN values exist
        print("Null values in each column:\n", self.df.isnull().sum())
        print(f"Does the dataset contain any null values? {self.df.isnull().values.any()}")

        
        # Return only rows from analysis start date forward
        return self.df.loc[self.analysis_start_date:]
