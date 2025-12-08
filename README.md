# ft_linear_regression

## Project Overview

This project implements a **simple linear regression** with a single feature to predict car prices based on mileage.

The goal is to create two programs:
1. **Training program** - Trains the model using gradient descent
2. **Prediction program** - Uses the trained model to predict prices

---

## Subject Requirements

You will implement a simple linear regression with a single feature - in this case, the mileage of the car.

### Two Programs

#### 1. Prediction Program (`prediction.py`)
- Prompts the user for a car's mileage
- Uses the trained model to estimate the price
- Formula: `estimatePrice(mileage) = θ0 + (θ1 * mileage)`
- Initially, theta0 and theta1 are set to 0

#### 2. Training Program (`training.py`)
- Reads the dataset from `data/data.csv`
- Performs linear regression using gradient descent
- Saves the learned parameters (theta0 and theta1) to `model.json`
- Uses the following update formulas:

```
tmpθ0 = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])
tmpθ1 = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i]) × mileage[i]
```

Where `m` is the number of training examples.

**Important:** θ0 and θ1 must be updated simultaneously.

---

## Bonus Features

✓ **Data Visualization** - Plot the data distribution  
✓ **Regression Line Visualization** - Plot the fitted line on the same graph  
✓ **Precision Metrics** - Calculate and display model performance metrics:
- **R² Score** - Coefficient of determination (% of variance explained)
- **MAE** - Mean Absolute Error in euros
- **RMSE** - Root Mean Squared Error
- **MAPE** - Mean Absolute Percentage Error
- **Convergence Analysis** - Cost reduction percentage and interpretation

---

## Installation

### Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Train the Model

Run the training program from the `src/` directory:

```bash
cd src/
python training.py
```

**What it does:**
- Loads the dataset from `../data/data.csv`
- Trains the model using gradient descent (1000 iterations, learning_rate=0.1)
- Displays training results:
  - Trained parameters (theta0, theta1)
  - Precision metrics (R², MAE, RMSE, MAPE)
  - Convergence analysis
- Generates visualizations:
  - **Left plot**: Actual data points + fitted regression line
  - **Right plot**: Cost function convergence over iterations
- Saves the model to `model.json`

### Step 2: Make Predictions

Run the prediction program:

```bash
python prediction.py
```

**What it does:**
- Loads the trained model from `model.json`
- Prompts: "Enter the mileage of the car: "
- Calculates and displays the estimated price
- Validates input (must be numeric)

**Example:**
```
Enter the mileage of the car: 150000
Predicted price: €4,850
```

---

## Project Structure

```
ft_linear_regression/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   └── data.csv             # Training dataset (km, price)
└── src/
    ├── training.py          # Main training program
    ├── prediction.py        # Main prediction program
    ├── training_utils.py    # ML algorithms (gradient descent, etc.)
    ├── csv_utils.py         # CSV file loading utilities
    ├── visualisation.py     # Plotting and metrics functions
    └── model.json           # Saved model parameters
```

---

## How It Works

### Linear Regression Formula

The model predicts price using:

**estimatePrice = θ0 + θ1 × mileage**

Where:
- **θ0** = Intercept (base price)
- **θ1** = Slope (price change per km)

### Gradient Descent Training

The algorithm iteratively updates θ0 and θ1 to minimize the cost function (MSE).

### Data Normalization

Mileage is normalized to [0, 1] range for numerical stability:

**km_normalized = (km - 22899) / (240000 - 22899)**

---

## Output Interpretation

### Training Output Example
```
CSV file loaded successfully.
Training model...
✓ Training complete!
theta0 (intercept): 7990.866013
theta1 (slope): -4610.205110
✓ Model saved to model.json

==================================================
MODEL PRECISION METRICS
==================================================

R² Score: 0.7329 (73.29%)
   → Explains 73% of price variance

Mean Absolute Error (MAE): €556.20
   → Average prediction error

Root Mean Squared Error (RMSE): €667.66
   → Penalizes large errors more

Mean Absolute Percentage Error (MAPE): 9.61%
   → Average error as percentage of price

==================================================

CONVERGENCE ANALYSIS
==================================================
Initial Cost (MSE): €3,500,000.00
Final Cost (MSE): €445,000.00
Cost Reduction: 87.29%
✓ Excellent convergence! The model learned very well.
```

### Prediction Output Example
```
Enter the mileage of the car: 100000
Predicted price: €5,250
```

---

## Hyperparameters

You can modify these in `training.py`:

- **learning_rate** (default: 0.1) - Step size for gradient descent
- **n_iterations** (default: 1000) - Number of training iterations

Lower learning rates train slower but more stably; higher values risk divergence.

---

## Model Evaluation

- **R² > 0.7**: Good model
- **R² > 0.5**: Acceptable model
- **R² < 0.3**: Poor model
- **MAPE < 10%**: Excellent predictions
- **MAPE < 20%**: Good predictions

---

## Color Coding

- **Green (✓)**: Success messages and good results
- **White**: Data values and metrics
- **Red (✗)**: Errors and issues

