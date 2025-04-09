from colorama import Fore, init
import pandas as pd
from os import access, R_OK

init(autoreset=True)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(Fore.GREEN + "CSV file loaded successfully.")
        return df

    except IOError:
        if not access(file_path, R_OK):
            print(Fore.RED + f'{file_path} is not readable.')
        else:
            print(Fore.RED + f"{file_path} is not a valid CSV file.")
        return None
    except pd.errors.ParserError:
        print(Fore.RED + f"{file_path} could not be parsed.")
        return None
    except pd.errors.EmptyDataError:
        print(Fore.RED + f"{file_path} is empty.")
        return None
    except FileNotFoundError:
        print(Fore.RED + f"{file_path} not found.")
        return None
    except Exception as e:
        print(Fore.RED + f"{type(e).__name__}: {e}")
        return None
