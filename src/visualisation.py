import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore


def plot_all_visualizations(mileage, price, cost_history, theta0, theta1):
    """
    Plot all visualizations (data + regression line + cost history) in a single window.
    
    Args:
        mileage (np.array): Original mileage values.
        price (np.array): Actual price values.
        cost_history (np.array): Cost values at each iteration.
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.
    """
    # Print convergence interpretation BEFORE displaying the graph
    print(f"\n{Fore.CYAN}CONVERGENCE ANALYSIS")
    print("="*50)
    initial_cost = cost_history[0]
    final_cost = cost_history[-1]
    cost_reduction = ((initial_cost - final_cost) / initial_cost) * 100
    
    print(f"Initial Cost (MSE): €{initial_cost:.2f}")
    print(f"Final Cost (MSE): €{final_cost:.2f}")
    print(f"Cost Reduction: {cost_reduction:.2f}%")
    
    if cost_reduction > 80:
        print(f"{Fore.GREEN}✓ Excellent convergence! The model learned very well.")
    elif cost_reduction > 50:
        print(f"{Fore.GREEN}✓ Good convergence. The model learned adequately.")
    elif cost_reduction > 20:
        print(f"△ Moderate convergence. The model could improve.")
    else:
        print(f"{Fore.RED}✗ Poor convergence. Consider adjusting learning_rate or iterations.")
    
    print("="*50 + "\n")
    
    # Now display the graphs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Data and regression line
    ax1 = axes[0]
    ax1.scatter(mileage, price, color='blue', label='Actual Data', alpha=0.6, s=50)
    
    x_line = np.linspace(np.min(mileage), np.max(mileage), 100)
    x_normalized = (x_line - 22899) / (240000 - 22899)
    y_line = theta0 + theta1 * x_normalized
    ax1.plot(x_line, y_line, color='red', linewidth=2, label='Linear Regression Line')
    
    ax1.set_xlabel('Mileage (km)', fontsize=12)
    ax1.set_ylabel('Price (€)', fontsize=12)
    ax1.set_title('Car Price Linear Regression', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cost history (convergence)
    ax2 = axes[1]
    ax2.plot(cost_history, linewidth=2, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost (MSE)', fontsize=12)
    ax2.set_title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_data_and_regression(mileage, price, theta0, theta1):
    """
    Plot the actual data points and the linear regression line.
    
    Args:
        mileage (np.array): Original mileage values.
        price (np.array): Actual price values.
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot actual data points
    plt.scatter(mileage, price, color='blue', label='Actual Data', alpha=0.6, s=50)
    
    # Plot regression line
    x_line = np.linspace(np.min(mileage), np.max(mileage), 100)
    x_normalized = (x_line - 22899) / (240000 - 22899)
    y_line = theta0 + theta1 * x_normalized
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Linear Regression Line')
    
    plt.xlabel('Mileage (km)', fontsize=12)
    plt.ylabel('Price (€)', fontsize=12)
    plt.title('Car Price Linear Regression', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cost_history(cost_history):
    """
    Plot the cost function over iterations to visualize convergence.
    
    Args:
        cost_history (np.array): Cost values at each iteration.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2, color='green')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_precision(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) to measure model precision.
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
    
    Returns:
        float: R² score (0 to 1, higher is better).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    return r2


def display_precision_metrics(y_true, y_pred):
    """
    Calculate and display multiple precision metrics.
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
    """
    # R² Score, pourcentage of variance explained by the model (0 to 1)
    r2 = calculate_precision(y_true, y_pred)
    
    # Mean Absolute Error (MAE), average absolute difference between true and predicted values
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error (RMSE), square root of average squared differences between true and predicted values
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Percentage Error (MAPE), average absolute percentage difference between true and predicted values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n" + "="*50)
    print("MODEL PRECISION METRICS")
    print("="*50)
    
    print(f"\nR² Score (Coefficient of Determination): {r2:.4f} ({r2*100:.2f}%)")
    print("   → Measures how much variance the model explains (closer to 1 = better)")
    
    print(f"\nMean Absolute Error (MAE): €{mae:.2f}")
    print("   → Average error in euros (how much predictions typically differ)")
    
    print(f"\nRoot Mean Squared Error (RMSE): €{rmse:.2f}")
    print("   → Like MAE, but penalizes large errors more heavily")
    
    print(f"\nMean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("   → Average error as a percentage of the actual price")
    
    print("\n" + "="*50 + "\n")
    
    return r2
