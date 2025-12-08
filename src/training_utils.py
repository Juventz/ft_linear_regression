import json
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore


def normalize_features(mileage):
    """
    Normalize mileage using min-max scaling.
    
    Args:
        mileage (np.array): Array of mileage values.
    
    Returns:
        tuple: (normalized_mileage, min_value, max_value)
    """
    min_val = np.min(mileage)
    max_val = np.max(mileage)
    normalized = (mileage - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def prepare_data(df):
    """
    Prepare data for training by extracting and normalizing features.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'km' and 'price'.
    
    Returns:
        tuple: (X, Y, normalization_params) where X is feature matrix with bias,
               Y is price vector, and normalization_params contains min/max values.
    """
    mileage = np.array(df.iloc[:, 0].values, dtype=float).reshape(-1, 1)
    price = np.array(df.iloc[:, 1].values, dtype=float).reshape(-1, 1)
    
    normalized_mileage, min_val, max_val = normalize_features(mileage)
    
    # Add bias term (column of ones)
    X = np.hstack((normalized_mileage, np.ones((normalized_mileage.shape[0], 1))))
    
    return X, price, {'min': min_val, 'max': max_val}


def model(X, theta):
    """
    Linear model: f(X) = X · θ
    
    Args:
        X (np.array): Feature matrix with bias term.
        theta (np.array): Parameters [theta1, theta0].
    
    Returns:
        np.array: Predictions.
    """
    return X.dot(theta)


def cost_function(X, y, theta):
    """
    Mean Squared Error (MSE) cost function: J(θ) = (1/2m)Σ(hθ(x) - y)²
    
    Args:
        X (np.array): Feature matrix.
        y (np.array): True values.
        theta (np.array): Parameters.
    
    Returns:
        float: Cost value.
    """
    m = len(y)
    predictions = model(X, theta)
    return 1 / (2 * m) * np.sum((predictions - y) ** 2)


def gradient(X, y, theta):
    """
    Compute gradient of cost function with respect to theta.
    
    Args:
        X (np.array): Feature matrix.
        y (np.array): True values.
        theta (np.array): Parameters.
    
    Returns:
        np.array: Gradient vector.
    """
    m = len(y)
    predictions = model(X, theta) - y
    return 1 / m * X.T.dot(predictions)


def gradient_descent(X, y, theta, learning_rate=0.1, n_iterations=1000):
    """
    Perform gradient descent to find optimal theta.
    
    Args:
        X (np.array): Feature matrix.
        y (np.array): True values.
        theta (np.array): Initial parameters.
        learning_rate (float): Learning rate.
        n_iterations (int): Number of iterations.
    
    Returns:
        tuple: (final_theta, cost_history)
    """
    cost_history = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        theta = theta - learning_rate * gradient(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history


def r_squared(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    Measures how well the model explains the variance in the data (0 to 1).
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
    
    Returns:
        float: R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)


def save_model(theta, norm_params):
    """
    Save trained model parameters and normalization parameters to JSON.
    
    Args:
        theta (np.array): Trained parameters [theta1, theta0].
        norm_params (dict): Normalization parameters {'min': ..., 'max': ...}.
    """
    model_data = {
        'theta0': float(theta[1, 0]),
        'theta1': float(theta[0, 0]),
        'normalization': {
            'min': float(norm_params['min']),
            'max': float(norm_params['max'])
        }
    }
    
    with open('model.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"{Fore.GREEN}✓ Model saved to model.json")


def plot_results(mileage, price, predictions):
    """
    Plot actual vs predicted prices.
    
    Args:
        mileage (np.array): Mileage values.
        price (np.array): Actual prices.
        predictions (np.array): Predicted prices.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, color='blue', label='Actual', alpha=0.6)
    plt.scatter(mileage, predictions, color='red', label='Predicted', alpha=0.6)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.title('Linear Regression: Car Price Prediction')
    plt.grid(True, alpha=0.3)
    plt.show()
