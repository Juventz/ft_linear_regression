import json

def predict_price(mileage, theta0, theta1):
    """
    Predict the price of a car based on its mileage using a linear regression model.
    
    Parameters:
    mileage (float): The mileage of the car.
    theta0 (float): The intercept of the linear regression model.
    theta1 (float): The slope of the linear regression model.
    
    Returns:
    float: The predicted price of the car.
    """
    km = (float(mileage) - 22899) / (240000 - 22899) # Normalize mileage
    price = theta0 + theta1 * km
    return round(price)