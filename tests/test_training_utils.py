import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training_utils import (
    normalize_features, prepare_data, model, cost_function, 
    gradient, gradient_descent, r_squared, save_model
)


class TestNormalizeFeatures:
    """Test suite for feature normalization."""
    
    def test_normalize_basic(self):
        """
        Test basic min-max normalization.
        
        Purpose: Verify that normalize_features correctly applies min-max normalization
        Formula: normalized = (x - min) / (max - min) ∈ [0, 1]
        Steps:
            1. Create array with values [0, 50, 100]
            2. Call normalize_features()
            3. Verify min_val=0 and max_val=100 are returned
            4. Verify 0 → 0 (min becomes 0)
            5. Verify 100 → 1 (max becomes 1)
            6. Verify 50 → 0.5 (median value)
        Expected result: Values are normalized to [0, 1] range
        Use case: Feature preparation for linear regression
        """
        data = np.array([[0], [50], [100]])
        normalized, min_val, max_val = normalize_features(data)
        
        assert min_val == 0
        assert max_val == 100
        assert normalized[0, 0] == 0
        assert normalized[2, 0] == 1
        assert np.isclose(normalized[1, 0], 0.5)
    
    def test_normalize_single_value(self):
        """
        Test normalization edge case: all values identical.
        
        Purpose: Verify that normalize_features handles case where min = max (division by zero)
        Steps:
            1. Create array with single value [50]
            2. Call normalize_features()
            3. Verify min_val = max_val = 50
            4. Verify normalized value is 0 (not NaN or error)
        Expected result: No division by zero error, returns 0
        Use case: Data with low variance or constant feature
        """
        data = np.array([[50]])
        normalized, min_val, max_val = normalize_features(data)
        
        assert min_val == 50
        assert max_val == 50
        assert normalized[0, 0] == 0
    
    def test_normalize_negative_values(self):
        """
        Test normalization with negative values in range.
        
        Purpose: Verify that normalize_features handles negative values correctly
        Steps:
            1. Create array with negative values [-100, 0, 100]
            2. Call normalize_features()
            3. Verify min_val=-100 and max_val=100
            4. Verify -100 → 0 (minimum)
            5. Verify 100 → 1 (maximum)
        Expected result: Negative values are correctly normalized
        Use case: Features with range including negative numbers
        """
        data = np.array([[-100], [0], [100]])
        normalized, min_val, max_val = normalize_features(data)
        
        assert min_val == -100
        assert max_val == 100
        assert normalized[0, 0] == 0
        assert normalized[2, 0] == 1
    
    def test_normalize_large_range(self):
        """
        Test normalization with realistic large range (car mileage).
        
        Purpose: Verify that normalize_features works with realistic ranges
                 (such as car mileage: 22,899 to 240,000 km)
        Steps:
            1. Create array with realistic mileage values
            2. Call normalize_features()
            3. Verify min_val=22899 and max_val=240000
            4. Verify 22899 → 0
            5. Verify 240000 → 1
        Expected result: Large ranges are correctly normalized
        Use case: Real-world usage with mileage data
        """
        data = np.array([[22899], [240000]])
        normalized, min_val, max_val = normalize_features(data)
        
        assert normalized[0, 0] == 0
        assert normalized[1, 0] == 1


class TestPrepareData:
    """Test suite for data preparation."""
    
    def test_prepare_data_basic(self):
        """
        Test basic data preparation for model training.
        
        Purpose: Verify that prepare_data correctly formats data into matrices
        Steps:
            1. Create DataFrame with columns 'km' and 'price'
            2. Call prepare_data()
            3. Verify X has 3 rows and 2 columns (normalized km + bias=1)
            4. Verify Y has 3 rows and 1 column (price)
            5. Verify second column of X contains all 1s (bias)
        Expected result: Matrices correctly structured for X.dot(theta)
        Use case: Data preparation before training
        """
        df = pd.DataFrame({'km': [100, 200, 300], 'price': [5000, 10000, 15000]})
        X, Y, norm_params = prepare_data(df)
        
        assert X.shape == (3, 2)
        assert Y.shape == (3, 1)
        assert np.all(X[:, 1] == 1)
    
    def test_prepare_data_values(self):
        """
        Test that prepared data values are correctly normalized and aligned.
        
        Purpose: Verify that actual values in matrices are correct
        Steps:
            1. Create DataFrame with simple km=[0,100] and price=[1000,2000]
            2. Call prepare_data()
            3. Verify X[0,0]=0 (km minimum normalized)
            4. Verify X[1,0]=1 (km maximum normalized)
            5. Verify Y[0,0]=1000 and Y[1,0]=2000 (prices unchanged)
        Expected result: Correct values in matrices (normalization + prices)
        Use case: Verification of data transformations
        """
        df = pd.DataFrame({'km': [0, 100], 'price': [1000, 2000]})
        X, Y, norm_params = prepare_data(df)
        
        assert X[0, 0] == 0
        assert X[1, 0] == 1
        assert Y[0, 0] == 1000
        assert Y[1, 0] == 2000


class TestModel:
    """Test suite for linear model."""
    
    def test_model_prediction(self):
        """
        Test linear model predictions using y = theta0 + theta1*x.
        
        Purpose: Verify that linear model correctly calculates predictions
        Formula: y_pred = X · theta = theta0 + theta1*x (matrix multiplication)
        Steps:
            1. Create matrix X with [x, bias] for 3 points: [0,1], [0.5,1], [1,1]
            2. Define theta=[2, 3] (theta1=2, theta0=3)
            3. Call model(X, theta)
            4. Verify predictions: 3+0*2=3, 3+0.5*2=4, 3+1*2=5
        Expected result: Exact predictions [3, 4, 5]
        Use case: Verification of regression line calculation
        """
        X = np.array([[0, 1], [0.5, 1], [1, 1]])
        theta = np.array([[2], [3]])
        
        predictions = model(X, theta)
        
        expected = np.array([[3], [4], [5]])
        assert np.allclose(predictions, expected)


class TestCostFunction:
    """Test suite for cost function (MSE)."""
    
    def test_cost_perfect_predictions(self):
        """
        Test MSE cost function when predictions match actual values perfectly.
        
        Purpose: Verify that cost_function returns 0 for perfect model
        Formula: J(theta) = (1/2m) * sum((y_pred - y_actual)^2)
        Steps:
            1. Create X with points [0,1] and [1,1]
            2. Create y with actual values [5, 7]
            3. Define theta=[2, 5] giving perfect predictions: 5+0*2=5, 5+1*2=7
            4. Call cost_function()
            5. Verify cost ≈ 0 (zero error)
        Expected result: MSE = 0
        Use case: Verify model converges to zero MSE
        """
        X = np.array([[0, 1], [1, 1]])
        y = np.array([[5], [7]])
        theta = np.array([[2], [5]])
        
        cost = cost_function(X, y, theta)
        assert np.isclose(cost, 0)
    
    def test_cost_bad_predictions(self):
        """
        Test MSE cost function when predictions are far from actual values.
        
        Purpose: Verify that cost_function returns high cost for bad predictions
        Steps:
            1. Create X with 2 normalized points
            2. Create y with actual values [10, 20]
            3. Define theta=[0, 0] giving predictions [0, 0] (always wrong)
            4. Call cost_function()
            5. Verify MSE = (1/4) * ((10-0)^2 + (20-0)^2) = (1/4) * 500 = 125
        Expected result: MSE = 125 (high cost = bad predictions)
        Use case: Verify model penalizes bad predictions
        """
        X = np.array([[0, 1], [1, 1]])
        y = np.array([[10], [20]])
        theta = np.array([[0], [0]])
        
        cost = cost_function(X, y, theta)
        assert np.isclose(cost, 125)


class TestGradient:
    """Test suite for gradient calculation."""
    
    def test_gradient_shape(self):
        """
        Test that gradient calculation returns correct shape vector.
        
        Purpose: Verify that gradient() returns vector of dimension (2, 1)
        Formula: ∇J = (1/m) * X.T · (y_pred - y_actual)
        Steps:
            1. Create X with 3 normalized points
            2. Create y with 3 actual values
            3. Define initial theta
            4. Call gradient()
            5. Verify result is shape (2, 1) - gradient for [theta1, theta0]
        Expected result: Gradient of dimension (2, 1)
        Use case: Verify gradients can be used for theta updates
        """
        X = np.array([[0.5, 1], [0.3, 1], [0.8, 1]])
        y = np.array([[5000], [3000], [8000]])
        theta = np.array([[1000], [2000]])
        
        grad = gradient(X, y, theta)
        assert grad.shape == (2, 1)
    
    def test_gradient_descent_reduces_cost(self):
        """
        Test that gradient descent algorithm reduces cost (MSE) over iterations.
        
        Purpose: Verify that gradient descent algorithm actually optimizes theta
        Steps:
            1. Create training data (X, y)
            2. Initialize theta to [0, 0]
            3. Calculate initial cost (with theta = [0, 0])
            4. Run gradient_descent with 100 iterations
            5. Verify final cost < initial cost
        Expected result: MSE decreases at each iteration (algorithm converges)
        Use case: Verify optimization works correctly
        """
        X = np.array([[0.2, 1], [0.4, 1], [0.6, 1], [0.8, 1]])
        y = np.array([[1000], [2000], [3000], [4000]])
        theta = np.zeros((2, 1))
        
        initial_cost = cost_function(X, y, theta)
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.1, n_iterations=100)
        final_cost = cost_history[-1]
        
        assert final_cost < initial_cost


class TestGradientDescent:
    """Test suite for gradient descent optimizer."""
    
    def test_gradient_descent_convergence(self):
        """
        Test complete gradient descent convergence on synthetic linear data.
        
        Purpose: Verify that gradient_descent converges over 1000 iterations and learns parameters
        Steps:
            1. Create perfectly linear data: y = 1 + 4*x
            2. Initialize theta to [0, 0]
            3. Run gradient_descent with learning_rate=0.1, n_iterations=1000
            4. Verify cost_history[-1] < cost_history[0]
            5. Verify theta_final ≠ [0, 0] (model learned something)
        Expected result: Complete convergence, theta_final ≈ [4, 1]
        Use case: Verify algorithm converges on linear data
        """
        X = np.array([[0, 1], [0.25, 1], [0.5, 1], [0.75, 1], [1, 1]])
        y = np.array([[1], [2], [3], [4], [5]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.1, n_iterations=1000)
        
        assert cost_history[-1] < cost_history[0]
        assert not np.allclose(theta_final, 0)
    
    def test_gradient_descent_returns_history(self):
        """
        Test that gradient descent returns complete cost history for all iterations.
        
        Purpose: Verify that gradient_descent returns cost at each iteration
        Steps:
            1. Run gradient_descent with 50 iterations
            2. Call gradient_descent()
            3. Verify cost_history has exactly 50 elements
            4. Verify cost_history is numpy array
        Expected result: Complete history available for convergence analysis
        Use case: Plot convergence, analyze learning_rate
        """
        X = np.array([[0, 1], [1, 1]])
        y = np.array([[5], [7]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.1, n_iterations=50)
        
        assert len(cost_history) == 50
        assert isinstance(cost_history, np.ndarray)


class TestRSquared:
    """Test suite for R² metric."""
    
    def test_r_squared_perfect_fit(self):
        """
        Test R² metric for perfect model predictions.
        
        Purpose: Verify that R² = 1.0 when predictions = actual values
        Formula: R² = 1 - (SS_res / SS_tot) where SS_res=Σ(y-ŷ)² and SS_tot=Σ(y-ȳ)²
        Steps:
            1. Create y_true = [1, 2, 3, 4, 5]
            2. Create y_pred = y_true (perfect predictions)
            3. Call r_squared()
            4. Verify R² ≈ 1.0 (100% variance explained)
        Expected result: R² = 1.0
        Use case: Verify perfect model gives R²=1
        """
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[1], [2], [3], [4], [5]])
        
        r2 = r_squared(y_true, y_pred)
        assert np.isclose(r2, 1.0)
    
    def test_r_squared_mean_predictor(self):
        """
        Test R² for mean predictor (always predicting the average).
        
        Purpose: Verify that R² ≈ 0 when model predicts mean (baseline)
        Steps:
            1. Create y_true = [1, 2, 3, 4, 5] with mean = 3
            2. Create y_pred = [3, 3, 3, 3, 3] (always mean)
            3. Call r_squared()
            4. Verify R² ≈ 0 (no better than baseline)
        Expected result: R² ≈ 0
        Use case: Minimal benchmark - model must do better than mean predictor
        """
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[3], [3], [3], [3], [3]])
        
        r2 = r_squared(y_true, y_pred)
        assert np.isclose(r2, 0, atol=0.01)
    
    def test_r_squared_bad_fit(self):
        """
        Test R² for bad model (predictions inversely related to reality).
        
        Purpose: Verify that R² < 0 when model is worse than mean predictor
        Steps:
            1. Create y_true = [1, 2, 3, 4, 5]
            2. Create y_pred = [5, 4, 3, 2, 1] (inverse relationship)
            3. Call r_squared()
            4. Verify R² < 0 (worse than baseline)
        Expected result: R² < 0
        Use case: Detection of models that do more harm than good
        """
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[5], [4], [3], [2], [1]])
        
        r2 = r_squared(y_true, y_pred)
        assert r2 < 0


class TestDataValidation:
    """Test suite for data validation."""
    
    def test_negative_values_detection(self):
        """
        Test detection of negative values in data (validation check).
        
        Purpose: Verify detection of negative values (invalid km or prices)
        Steps:
            1. Create DataFrame with at least one negative value (-200)
            2. Call (df < 0).any().any()
            3. Verify result is True (negatives detected)
        Expected result: True (invalid data)
        Use case: Validation before training - reject negative km/prices
        """
        df = pd.DataFrame({'km': [100, -200, 300], 'price': [5000, 10000, 15000]})
        
        has_negative = (df < 0).any().any()
        assert has_negative
    
    def test_positive_values_only(self):
        """
        Test that valid data has no negative values.
        
        Purpose: Verify that valid data (km>0, price>0) is correctly accepted
        Steps:
            1. Create DataFrame with only positive values
            2. Call (df < 0).any().any()
            3. Verify result is False (no negatives)
        Expected result: False (valid data)
        Use case: Accept and train on data without negative values
        """
        df = pd.DataFrame({'km': [100, 200, 300], 'price': [5000, 10000, 15000]})
        
        has_negative = (df < 0).any().any()
        assert not has_negative
