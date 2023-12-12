import numpy as np
import pandas as pd
import json

def is_2dig_naics(naics_code):
    """
    Determines if the first 2 characters in a given string is a 2-digit NAICS code

    Parameters
    ----------
    naics_code: str
        The input string to evaluate.
    
    Returns
    -------
        boolean
            Whether or not the first 2 chars of naics_code is a valid NAICS code.
    """
    return naics_code[:2].isnumeric() and not any(char.isdigit() for char in naics_code[2:])

def process_zbp_details(data, year, zip_codes):
    """
    Processes and standardizes a ZIP Code Industry Details dataset from the U.S. Census' County Business Patterns Data

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to process
    year: int
        The year that this data refers to
    zip_codes: List[int]
        The zip_codes to include in processed data

    Returns
    -------
    pandas.DataFrame
        The input data after being processed for:
        - filter out irrelavent ZIP Codes
        - standardize column names
        - filter out NaN values
        - convert 8 digit NAICS codes to 2 digit
        - determine and reinforce data types
        - removing definition columns
    """
    cols = ['zip', 'naics', 'est']
    # column 'n<5' change to 'n1_4' in 2016
    if year <= 2016:
        cols += ['n1_4']
    else:
        cols += ['n<5']
    cols += ['n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']
    data = data[cols]
    data = data.rename(columns={'n<5':'n1_4'})
    # process only zip-codes contained in config.json
    data = data[data['zip'].apply(lambda x: x in zip_codes)]
    data = data[data['naics'].apply(is_2dig_naics)]
    data['naics'] = data['naics'].apply(lambda x: x[:2])
    final_est_bin_cols = ['n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']
    for col in final_est_bin_cols:
        data[col] = data[col].apply(lambda x: x if x != 'N' else 0).astype('int64')
    data = data.assign(year=np.full(data.shape[0], year))
    return data

def process_zbp_totals(data, year, zip_codes):
    """
    Processes and standardizes a ZIP Code Totals dataset from the U.S. Census' County Business Patterns Data
    
    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to process
    year: int
        The year that this data refers to
    zip_codes: List[int]
        The zip_codes to include in processed data

    Returns
    -------
    pandas.DataFrame
        The input data after being processed for:
        - filter out irrelavent ZIP Codes
        - standardize column names
        - determine and reinforce data types
        - removing definition columns
    """
    cols_to_drop = ['name', 'city', 'stabbr', 'cty_name']
    # new descriptive column 'empflag' introduced to CBP after 2017
    # removed along with other descriptive columns 
    if year <= 2017:
        cols_to_drop += ['empflag']
    data = data.drop(columns=cols_to_drop)
    data = data[data['zip'].apply(lambda x: x in zip_codes)]
    data = data.assign(year=np.full(data.shape[0], year))
    return data


def run():
    """
    Processes and merges all ZIP Code Industry Details and Totals datasets in the data/raw/zbp_data directory.

    Out
    -----
    processed_zbp_detail_data: CSV file
        Combined processed data for all ZIP Code Industry Details data 
    processed_zbp_totals_data: CSV file
        Combined processed data for all ZIP Code Totals data 
    """
    with open('config/config.json', 'r') as fh:
        params = json.load(fh)

    # LOAD DATA
    zbp_detail_by_year = {}
    zbp_totals_by_year = {}
    for year in params['years']:
        shortened_year = str(year)[2:]
        detail_encoding = None
        totals_encoding = None
        if year >= 2017:
            detail_encoding = 'latin-1'
        if year >= 2018:
            totals_encoding = 'latin-1'
        zbp_detail_by_year[year] = pd.read_csv(f'src/data/raw/zbp_data/zbpdetail/zbp{shortened_year}detail/zbp{shortened_year}detail.txt', encoding=detail_encoding)
        zbp_totals_by_year[year] = pd.read_csv(f'src/data/raw/zbp_data/zbptotals/zbp{shortened_year}totals/zbp{shortened_year}totals.txt', encoding=totals_encoding)

    # PROCESS AND CONCATENATE DATA
    zip_codes = params['zip_codes']

    for year in zbp_detail_by_year:
        zbp_detail_by_year[year] = process_zbp_details(zbp_detail_by_year[year], year, zip_codes)

    for year in zbp_totals_by_year:
        zbp_totals_by_year[year] = process_zbp_totals(zbp_totals_by_year[year], year, zip_codes)
        
    # SAVE PROCESSED DATA
    zbp_detail_data = pd.concat(list(zbp_detail_by_year.values()), ignore_index=True).reset_index(drop=True)
    zbp_detail_data.to_csv('src/data/temp/processed_zbp_detail_data.csv', index=False)

    zbp_totals_data = pd.concat(list(zbp_totals_by_year.values()), ignore_index=True).reset_index(drop=True)
    zbp_totals_data.to_csv('src/data/temp/processed_zbp_totals_data.csv', index=False)


if __name__ == '__main__':
    run()