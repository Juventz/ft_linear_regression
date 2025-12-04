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


def main():
    try:
        with open('model.json', 'r') as f:
            model = json.load(f)
        
        theta0 = model['theta0']
        theta1 = model['theta1']
        
        while(42):
            mileage = input("Enter the mileage of the car: ")
            if not mileage:
                print("Mileage input cannot be empty. Please try again.")
            else:
                try:
                    assert(all(char.isdigit() for char in mileage)), "Mileage must be a numeric value."
                    break
                except AssertionError as error:
                    print(f"Input Error: {error}")
    
        predicted_price = predict_price(mileage, theta0, theta1)
        if predicted_price <= 0:
            print("The predicted price is negative or null, which is not valid.")
        else:
            print(f"Predicted price: {predicted_price}")
            return 

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return  
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return


if __name__ == "__main__":
    main()