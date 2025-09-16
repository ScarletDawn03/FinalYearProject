import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error

class EvaluationMetrics:
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred) * 100 # Not used in this research but useful metric

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def accuracy(y_true, y_pred, threshold_percent=5):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_true = np.where(y_true == 0, 1e-8, y_true)
        percentage_diff = np.abs((y_pred - y_true) / y_true) * 100
        return (np.sum(percentage_diff <= threshold_percent) / len(y_true)) * 100
    
    @staticmethod
    def profitability_index(y_true, y_pred, forecast_window, initial_capital=10000.0):
        capital = initial_capital
        position = 0
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        for i in range(len(y_pred) - forecast_window):
            current_price = y_true[i]
            predicted_future_price = y_pred[i]

            if predicted_future_price > current_price and position == 0:
                position = capital / current_price
                capital = 0
            elif predicted_future_price < current_price and position > 0:
                capital = position * current_price
                position = 0

        if position > 0:
            capital = position * y_true[-1]

        return capital / initial_capital

