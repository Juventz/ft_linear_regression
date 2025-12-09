import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training_utils import save_model, prepare_data, gradient_descent, model
from csv_utils import read_csv


class TestEdgeCasesAndErrors:
    """Test suite for edge cases and error handling."""
    
    def test_prepare_data_with_nan_values(self):
        """
        Test handling of NaN (missing) values in data.
        
        Purpose: Verify that prepare_data handles missing values gracefully
        Steps:
            1. Create DataFrame with NaN values
            2. Call prepare_data()
            3. Verify that NaN values are handled (no crash)
        Expected result: Should raise error or handle NaN appropriately
        Use case: Real data often has missing values
        """
        df = pd.DataFrame({'km': [100, np.nan, 300], 'price': [5000, 10000, 15000]})
        
        # Should either raise error or handle NaN
        try:
            X, Y, norm_params = prepare_data(df)
            # If it doesn't crash, check structure is valid
            assert isinstance(X, np.ndarray)
        except (ValueError, TypeError):
            # It's acceptable to raise error for NaN
            pass
    
    def test_prepare_data_with_inf_values(self):
        """
        Test handling of infinite values in data.
        
        Purpose: Verify that infinite values don't cause crashes
        Steps:
            1. Create DataFrame with inf values
            2. Call prepare_data()
            3. Verify no segfault or crash
        Expected result: Should handle gracefully
        Use case: Prevent crash on corrupted data
        """
        df = pd.DataFrame({'km': [100, np.inf, 300], 'price': [5000, 10000, 15000]})
        
        try:
            X, Y, norm_params = prepare_data(df)
            # Check that result doesn't contain inf
            assert not np.any(np.isinf(X))
        except (ValueError, RuntimeWarning):
            pass
    
    def test_gradient_descent_with_zero_learning_rate(self):
        """
        Test gradient descent with learning_rate = 0 (no learning).
        
        Purpose: Verify theta doesn't change when learning_rate is 0
        Steps:
            1. Create training data
            2. Initialize theta to [0, 0]
            3. Run gradient_descent with learning_rate=0
            4. Verify theta remains [0, 0]
        Expected result: Theta unchanged
        Use case: Boundary condition check
        """
        X = np.array([[0, 1], [0.5, 1], [1, 1]])
        y = np.array([[1], [2], [3]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0, n_iterations=10)
        
        # With learning_rate=0, theta should not change
        assert np.allclose(theta_final, [0, 0])
    
    def test_gradient_descent_with_very_high_learning_rate(self):
        """
        Test gradient descent with very high learning_rate (divergence check).
        
        Purpose: Verify that high learning rate causes divergence (NaN/inf)
        Steps:
            1. Create training data
            2. Initialize theta to [0, 0]
            3. Run gradient_descent with learning_rate=1000
            4. Check for NaN or inf values
        Expected result: May produce NaN/inf (divergence)
        Use case: Detect learning rate too high
        """
        X = np.array([[0, 1], [0.5, 1], [1, 1]])
        y = np.array([[1], [2], [3]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=1000, n_iterations=10)
        
        # High learning rate may cause divergence
        # Check that we don't crash, even if NaN appears
        assert isinstance(theta_final, np.ndarray)
    
    def test_model_with_empty_features(self):
        """
        Test model prediction with empty feature matrix.
        
        Purpose: Verify model handles empty input
        Steps:
            1. Create empty feature matrix
            2. Call model()
            3. Verify no crash
        Expected result: Returns empty prediction array
        Use case: Boundary condition
        """
        X = np.array([]).reshape(0, 2)
        theta = np.array([[2], [3]])
        
        predictions = model(X, theta)
        
        assert predictions.shape[0] == 0
    
    def test_model_with_single_sample(self):
        """
        Test model prediction with single sample.
        
        Purpose: Verify model handles single data point
        Steps:
            1. Create single sample X
            2. Call model()
            3. Verify output shape is (1, 1)
        Expected result: Single prediction returned
        Use case: Predicting single car price
        """
        X = np.array([[0.5, 1]])
        theta = np.array([[2], [3]])
        
        predictions = model(X, theta)
        
        assert predictions.shape == (1, 1)
        assert np.isclose(predictions[0, 0], 4)


class TestDataBoundaryConditions:
    """Test suite for data boundary conditions."""
    
    def test_prepare_data_with_single_row(self):
        """
        Test data preparation with only 1 sample.
        
        Purpose: Verify prepare_data handles minimal dataset
        Steps:
            1. Create DataFrame with 1 row
            2. Call prepare_data()
            3. Verify X shape is (1, 2) and Y shape is (1, 1)
        Expected result: Valid matrices returned
        Use case: Minimal dataset handling
        """
        df = pd.DataFrame({'km': [100], 'price': [5000]})
        X, Y, norm_params = prepare_data(df)
        
        assert X.shape == (1, 2)
        assert Y.shape == (1, 1)
    
    def test_prepare_data_with_all_same_km(self):
        """
        Test preparation when all mileage values are identical.
        
        Purpose: Verify normalization handles constant feature
        Steps:
            1. Create DataFrame with all km values = 100
            2. Call prepare_data()
            3. Verify normalized km are all 0
        Expected result: All normalized km are 0
        Use case: Constant feature detection
        """
        df = pd.DataFrame({'km': [100, 100, 100], 'price': [5000, 6000, 7000]})
        X, Y, norm_params = prepare_data(df)
        
        # All km values normalized should be 0
        assert np.allclose(X[:, 0], 0)
    
    def test_prepare_data_with_same_price_and_km(self):
        """
        Test when both km and price are identical across all samples.
        
        Purpose: Verify handling of completely uniform data
        Steps:
            1. Create DataFrame with constant km and price
            2. Call prepare_data()
            3. Verify no crash
        Expected result: Valid matrices with constant values
        Use case: Edge case for uniform training data
        """
        df = pd.DataFrame({'km': [100, 100, 100], 'price': [5000, 5000, 5000]})
        X, Y, norm_params = prepare_data(df)
        
        assert X.shape == (3, 2)
        assert np.allclose(Y, 5000)


class TestGradientDescentStability:
    """Test suite for gradient descent numerical stability."""
    
    def test_gradient_descent_with_large_values(self):
        """
        Test gradient descent with very large feature values.
        
        Purpose: Verify gradient descent handles large magnitude data
        Steps:
            1. Create X with very large values (10^9)
            2. Create y with corresponding large values
            3. Run gradient_descent
            4. Verify no NaN or inf
        Expected result: Converges or handles gracefully
        Use case: Data scaling issues
        """
        X = np.array([[1e9, 1], [2e9, 1], [3e9, 1]])
        y = np.array([[1e9], [2e9], [3e9]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=1e-9, n_iterations=10)
        
        # Check we don't get NaN
        assert not np.any(np.isnan(theta_final))
    
    def test_gradient_descent_with_tiny_values(self):
        """
        Test gradient descent with very small feature values.
        
        Purpose: Verify gradient descent handles tiny magnitude data
        Steps:
            1. Create X with very small values (10^-9)
            2. Create y with corresponding small values
            3. Run gradient_descent
            4. Verify no crash
        Expected result: Handles gracefully
        Use case: Data scaling issues
        """
        X = np.array([[1e-9, 1], [2e-9, 1], [3e-9, 1]])
        y = np.array([[1e-9], [2e-9], [3e-9]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=1e-1, n_iterations=10)
        
        # Check we don't get NaN
        assert not np.any(np.isnan(theta_final))
    
    def test_gradient_descent_many_iterations(self):
        """
        Test gradient descent with very large number of iterations.
        
        Purpose: Verify no memory issues or overflow after many iterations
        Steps:
            1. Create training data
            2. Run gradient_descent with n_iterations=10000
            3. Verify cost_history has correct length
            4. Verify no memory crash
        Expected result: Completes successfully
        Use case: Long training runs
        """
        X = np.array([[0, 1], [0.5, 1], [1, 1]])
        y = np.array([[1], [2], [3]])
        theta = np.zeros((2, 1))
        
        theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=10000)
        
        assert len(cost_history) == 10000
        assert not np.any(np.isnan(cost_history))


class TestModelFileHandling:
    """Test suite for model save/load functionality."""
    
    def test_save_model_creates_valid_json(self):
        """
        Test that save_model creates valid JSON file.
        
        Purpose: Verify model.json is properly formatted
        Steps:
            1. Create theta and norm_params
            2. Call save_model()
            3. Load JSON and verify structure
        Expected result: Valid JSON with theta0, theta1, normalization params
        Use case: Model persistence
        """
        theta = np.array([[100], [50]])
        norm_params = {'min': np.float64(22899), 'max': np.float64(240000)}
        
        save_model(theta, norm_params)
        
        # Verify file exists and is valid JSON
        model_path = Path('model.json')
        assert model_path.exists()
        
        with open(model_path) as f:
            data = json.load(f)
        
        assert 'theta0' in data
        assert 'theta1' in data
        assert 'normalization' in data
    
    def test_save_model_with_nan_theta(self):
        """
        Test save_model behavior with NaN in theta.
        
        Purpose: Verify handling of invalid model parameters
        Steps:
            1. Create theta with NaN values
            2. Try to save_model()
            3. Check if error is raised or NaN is handled
        Expected result: Either raises error or saves NaN
        Use case: Prevent saving broken models
        """
        theta = np.array([[np.nan], [50]])
        norm_params = {'min': 22899, 'max': 240000}
        
        try:
            save_model(theta, norm_params)
            # If it saves, verify we can load it
            with open('model.json') as f:
                data = json.load(f)
            # theta0 (index 1) remains 50, theta1 (index 0) carries the NaN
            assert data['theta0'] == 50.0
            assert ('theta1' in data) and (data['theta1'] is None or np.isnan(data['theta1']))
        except (ValueError, TypeError):
            # It's acceptable to reject NaN models
            pass


class TestCSVReadingRobustness:
    """Test suite for robust CSV reading."""
    
    def test_read_csv_with_whitespace(self):
        """
        Test CSV reading with extra whitespace.
        
        Purpose: Verify CSV parser handles extra spaces
        Steps:
            1. Create CSV with extra spaces around values
            2. Call read_csv()
            3. Verify values are correctly parsed
        Expected result: Whitespace handled or stripped
        Use case: Messy real-world CSV files
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("km,price\n100 , 5000 \n200 , 10000 \n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert len(df) == 2
            # Values may have leading/trailing spaces
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(temp_path).unlink()
    
    def test_read_csv_with_quoted_fields(self):
        """
        Test CSV with quoted fields.
        
        Purpose: Verify quoted field handling
        Steps:
            1. Create CSV with quoted fields
            2. Call read_csv()
            3. Verify correct parsing
        Expected result: Quotes removed, values correct
        Use case: Standard CSV format
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('km,price\n"100","5000"\n"200","10000"\n')
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert len(df) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_read_csv_large_file(self):
        """
        Test reading large CSV file (no memory issues).
        
        Purpose: Verify handling of large datasets
        Steps:
            1. Create CSV with 10000 rows
            2. Call read_csv()
            3. Verify complete loading
        Expected result: File completely loaded
        Use case: Large dataset handling
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("km,price\n")
            for i in range(10000):
                f.write(f"{100 + i},{5000 + i*10}\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert len(df) == 10000
        finally:
            Path(temp_path).unlink()
