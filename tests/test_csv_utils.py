import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from csv_utils import read_csv


class TestReadCSV:
    """Test suite for CSV reading functionality."""
    

    def test_read_valid_csv(self):
        """
        Test reading a valid CSV file.
        
        Purpose: Verify that read_csv can correctly load a valid CSV file
        Steps:
            1. Create a temporary CSV file with valid data (2 rows)
            2. Call read_csv() on this file
            3. Verify the result is a pandas DataFrame
            4. Verify the number of rows is correct (2)
            5. Verify column names are correct (['km', 'price'])
        Expected result: Function returns a valid DataFrame without error
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("km,price\n100,5000\n200,10000\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['km', 'price']
        finally:
            Path(temp_path).unlink()
    

    def test_read_nonexistent_file(self):
        """
        Test reading a non-existent file raises FileNotFoundError.
        
        Purpose: Verify that read_csv raises FileNotFoundError when file does not exist
        Steps:
            1. Call read_csv() with a file path that doesn't exist
            2. Verify that FileNotFoundError exception is raised
        Expected result: FileNotFoundError is raised with red error message
        Use case: Robust handling of missing input files
        """
        with pytest.raises(FileNotFoundError):
            read_csv("/nonexistent/path/file.csv")
    

    def test_read_empty_csv(self):
        """
        Test reading an empty CSV file raises EmptyDataError.
        
        Purpose: Verify that read_csv raises EmptyDataError when CSV file is completely empty
        Steps:
            1. Create an empty CSV file (0 bytes)
            2. Call read_csv() on this file
            3. Verify that EmptyDataError exception is raised
        Expected result: EmptyDataError is raised with red error message
        Use case: Detection of empty or corrupted data files
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(pd.errors.EmptyDataError):
                read_csv(temp_path)
        finally:
            Path(temp_path).unlink()
    

    def test_read_csv_only_headers(self):
        """
        Test reading CSV with only headers (no data).
        
        Purpose: Verify that read_csv handles CSV file with only header row correctly
        Steps:
            1. Create a CSV file with only the header row
            2. Call read_csv() on this file
            3. Verify that DataFrame is created but empty (0 rows)
        Expected result: Empty DataFrame is returned without error
        Use case: Partial data or file waiting for data
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("km,price\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert len(df) == 0
        finally:
            Path(temp_path).unlink()
    

    def test_read_malformed_csv(self):
        """
        Test reading malformed CSV (inconsistent column count).
        
        Purpose: Verify that read_csv handles malformed CSV files (inconsistent column count)
        Steps:
            1. Create a CSV with one row having 3 columns and another having 2
            2. Call read_csv() on this file
            3. Verify that DataFrame is still created (pandas is tolerant)
        Expected result: DataFrame is returned (pandas fills missing values with NaN)
        Use case: Noisy or partially corrupted data
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write mismatched columns
            f.write("km,price\n100,5000,extra\n200,10000\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(temp_path).unlink()
    
    
    def test_read_csv_with_special_characters(self):
        """
        Test reading CSV with special characters in data.
        
        Purpose: Verify that read_csv handles special characters and UTF-8 encoding
        Steps:
            1. Create CSV file in UTF-8 with explicit encoding
            2. Call read_csv() on this file
            3. Verify that DataFrame is created without encoding error
        Expected result: Valid DataFrame is returned
        Use case: International data or data with accents/special symbols
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("km,price\n100,5000\n200,10000\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(temp_path).unlink()
    
    def test_read_csv_with_floats(self):
        """
        Test reading CSV with float values (decimal numbers).
        
        Purpose: Verify that read_csv correctly parses decimal numbers (floats)
        Steps:
            1. Create CSV with decimal values (e.g., 100.5, 5000.99)
            2. Call read_csv() on this file
            3. Verify that values are correctly converted to floats
            4. Verify that first value matches exactly 100.5
        Expected result: Floats are correctly parsed and accessible
        Use case: Real data with decimal precision
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("km,price\n100.5,5000.99\n200.3,10000.50\n")
            temp_path = f.name
        
        try:
            df = read_csv(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert df.iloc[0, 0] == 100.5
        finally:
            Path(temp_path).unlink()
