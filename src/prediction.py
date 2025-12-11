import json
import re
from colorama import Fore, init

init(autoreset=True)


def predict_price(mileage, theta0, theta1, norm_params=None):
    """
    Predict the price of a car based on its mileage using a linear regression model.
    """
    norm = norm_params or {'min': 22899, 'max': 240000}
    min_km = norm.get('min', 22899)
    max_km = norm.get('max', 240000)
    if max_km == min_km:
        km = 0.0
    else:
        km = (float(mileage) - min_km) / (max_km - min_km)
    price = theta0 + theta1 * km
    return round(price)


def main():
    try:
        with open('model.json', 'r') as f:
            model = json.load(f)
        
        theta0 = model['theta0']
        theta1 = model['theta1']
        norm_params = model.get('normalization')
        
        pattern = re.compile(r"^[0-9]+(\.[0-9]+)?$")
        while True:
            mileage_input = input("Enter the mileage of the car: ").strip()
            if not mileage_input:
                print(f"{Fore.RED}Mileage input cannot be empty. Please try again.")
                continue
            if not pattern.match(mileage_input):
                print(f"{Fore.RED}Input Error: Mileage must be a positive integer or decimal.")
                continue
            mileage = float(mileage_input)
            break
    
        predicted_price = predict_price(mileage, theta0, theta1, norm_params)
        if predicted_price <= 0:
            print(f"{Fore.RED}The predicted price is negative or null, which is not valid.")
        else:
            print(f"{Fore.GREEN}Predicted price: €{predicted_price:,}")
            return 

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Process interrupted by user.")
        return  
    except FileNotFoundError:
        print(f"{Fore.RED}Error: model.json not found. Please train the model first by running training.py")
        return
    except Exception as e:
        print(f"{Fore.RED}{type(e).__name__}: {e}")
        return


if __name__ == "__main__":
    main()