import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def dummy_stock_data():
    date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Adj Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(1000, 10000, size=100)
    }, index=date_range)
