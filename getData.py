# Getting data from yahoo finance
import pandas as pd
import numpy as np
import yfinance as yf   
#from sklearn.preprocessing import MinMaxScaler

def download_data(start_number=0,end_number = 500, start_date='2010-01-01',end_date='2024-06-14'):

    #Stock_name= ['SPY','AAPL','MSFT','AMZN','EZJ','JNJ','SBUX','PFE','T']
    #original_setting = pd.options.mode.chained_assignment

    # Temporarily set it to 'raise' to suppress the warning
    #pd.options.mode.chained_assignment = None

    Stock_name =['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']


    final_data = {}
    for ticker in Stock_name[start_number:end_number]:
        historic_price = yf.download(ticker, start_date, end_date)
        final_data[ticker] = historic_price
    return final_data
   

print(download_data())