import numpy as np
import pandas as pd

def is_2dig_naics(naics_code):
    """
    Checks if a given NAICS code string is a 2-digit NAICS code

    Parameters
    ----------
    naics_code: str
        String representation of an NAICS code
        Expected to range from 2-digit to 5-digit representations 

    Returns
    -------
    boolean:
        Whether or not the input string is a valid 2-digit NAICS code

    Example
    -------
        >>> is_2dig_naics('236///')
        False
        >>> is_2dig_naics('23----')
        True
    """
    return naics_code[:2].isnumeric() and not any(char.isdigit() for char in naics_code[2:])

def process_zbp_data(data, year, params):
    """
    Processes a single ZBP Details dataset in src/data/raw/zbp_data/zbpdetail

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to processes
    year: int
        The year corresponding to the current dataset
    params: dict
        Python dict representation of config/config.json

    Returns
    -------
    pandas.DataFrame:
        Dataset with columns ['zip', 'naics', 'est', 'n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000', 'year']
    """
    cols = ['zip', 'naics', 'est']
    if year <= 2016:
        cols += ['n1_4']
    else:
        cols += ['n<5']
    cols += ['n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']
    
    # filter and standardize col names
    data = data[cols]
    data = data.rename(columns={'n<5':'n1_4'})
    # filter only relavent zip codes
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    # keep only 2dig naics
    data = data[data['naics'].apply(is_2dig_naics)]
    data['naics'] = data['naics'].apply(lambda x: x[:2])
    # cast est size bin cols to int, TREATING 'N'(Not available or not comparable) RECORDS AS 0
    for col in ['n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']:
        data[col] = data[col].apply(lambda x: x if x != 'N' else 0).astype('int64')
    # assign year variable
    data = data.assign(year=np.full(data.shape[0], year))
    
    return data