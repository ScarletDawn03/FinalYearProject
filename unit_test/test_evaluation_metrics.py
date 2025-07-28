import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReusableFunctions")))

from EvaluationMetrics import EvaluationMetrics 

def test_rmse():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    result = EvaluationMetrics.rmse(y_true, y_pred)
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(result, expected)

def test_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = EvaluationMetrics.mape(y_true, y_pred)
    expected = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    assert np.isclose(result, expected)

def test_r2():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert EvaluationMetrics.r2(y_true, y_pred) == 1.0

def test_accuracy():
    y_true = np.array([100, 105, 95])
    y_pred = np.array([104, 107, 93])  # All within 5%
    result = EvaluationMetrics.accuracy(y_true, y_pred, threshold_percent=5)
    assert result == 100.0

def test_accuracy_with_outliers():
    y_true = np.array([100, 100, 100])
    y_pred = np.array([90, 105, 130])  # Only 1 within 5%
    result = EvaluationMetrics.accuracy(y_true, y_pred, threshold_percent=5)
    assert np.isclose(result, 33.3333333333, atol=1e-2)

def test_profitability_index():
    y_true = np.array([100, 105, 110, 120, 115])
    y_pred = np.array([101, 106, 111, 125, 130])
    result = EvaluationMetrics.profitability_index(y_true, y_pred, forecast_window=1, initial_capital=10000)
    assert result > 1.0  # Because we would’ve bought low and sold higher in this setup
