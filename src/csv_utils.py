from colorama import Fore, init
import pandas as pd
from os import access, R_OK

init(autoreset=True)

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file is not readable.
        pd.errors.ParserError: If the CSV cannot be parsed.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    try:
        df = pd.read_csv(file_path)
        print(Fore.GREEN + "CSV file loaded successfully.")
        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(Fore.RED + f"{file_path} not found.") from e
    except IOError as e:
        if not access(file_path, R_OK):
            raise IOError(Fore.RED + f"{file_path} is not readable.") from e
        else:
            raise IOError(Fore.RED + f"{file_path} is not a valid CSV file.") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(Fore.RED + f"{file_path} is empty.") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(Fore.RED + f"{file_path} could not be parsed.") from e
    except Exception as e:
        raise Exception(Fore.RED + f"{type(e).__name__}: {e}") from e
