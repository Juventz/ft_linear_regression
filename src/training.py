from colorama import init, Fore 
import numpy as np
from csv_utils import read_csv
from training_utils import prepare_data, gradient_descent, model, r_squared, save_model, plot_results
from visualisation import plot_all_visualizations, display_precision_metrics

init(autoreset=True)


def main():
    """Main training pipeline."""
    try:
        df = read_csv('../data/data.csv')
        if df is None:
            return
        
        # Check for negative values in mileage and price
        if (df < 0).any().any():
            raise ValueError(Fore.RED + "Mileage and Price must be non-negative.")
        
        # Prepare data 
        X, Y, norm_params = prepare_data(df)
        
        # Initialize parameters
        theta = np.zeros((2, 1))
        
        # Train model
        print("Training model...")
        theta_final, cost_history = gradient_descent(
            X, Y, theta,
            learning_rate=0.1,
            n_iterations=1000
        )
        
        # Evaluate model
        predictions = model(X, theta_final)
        r2_score = r_squared(Y, predictions)
        
        print(f"{Fore.GREEN}✓ Training complete!")
        print(f"theta0 (intercept): {theta_final[1, 0]:.6f}")
        print(f"theta1 (slope): {theta_final[0, 0]:.6f}")
        
        # Save model
        save_model(theta_final, norm_params)
        
        # === BONUS: Visualizations and Precision Metrics ===
        
        # Display precision metrics
        display_precision_metrics(Y, predictions)
        
        # Plot all visualizations in one window
        mileage_original = np.array(df.iloc[:, 0].values, dtype=float)
        plot_all_visualizations(mileage_original, Y, cost_history, theta_final[1, 0], theta_final[0, 0])
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}✗ Training interrupted by user.")
        return
    
    except Exception as e:
        print(f"{Fore.RED}✗ {type(e).__name__}: {e}")
        return


if __name__ == "__main__":
    main()
