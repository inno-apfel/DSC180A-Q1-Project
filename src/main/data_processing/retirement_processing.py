import numpy as np
import pandas as pd


def process_retire_data(data, year, params):
    """
    Processes a single dataset in src/data/raw/retire_data.

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
        Dataset with columns ['zip', 'year', 'total_retirement', 'total_midcareer (25-34)', 'total_midcareer (35-44)']
    """
    
    column_names = data.columns
    # Only getting Estimates
    filtered_columns = [col for col in column_names if col[-8:] == 'Estimate' and col[6:11].isdigit()]
    data = data[filtered_columns]
    # Cleaning up so that the rows are just the zipcode
    data.columns = data.columns.str.replace('ZCTA5 ', '').str.replace('!!Estimate', '')
    # Selecting: Total population and retired aged
    selected_rows = data.loc[[1, 9, 10, 14, 15, 16]]
    # Define the new row names
    og_names = {
                1: 'Total population',
                9: '25 to 34 years',
                10: '35 to 44 years',
                14: '65 to 74 years',
                15: '75 to 84 years',
                16: '85 years and over'
               }

    data = selected_rows.rename(index=og_names)
    # Remove commas from all numbers in all columns
    data = data.map(lambda x: str(x).replace(',', ''))
    # Convert all columns to numeric type
    data = data.apply(pd.to_numeric, errors='ignore')
    # Transpose dataframe and convert zip-code col
    data = data.T
    data.index = data.index.set_names(['zip'])
    data = data.reset_index()
    data['zip'] = data['zip'].astype('int64')
    # Assign year variable
    data = data.assign(year=np.full(data.shape[0], year))
    # Keeping only relevant zip-codes
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    # retirement col
    data['total_retirement'] = data['65 to 74 years'] + data['75 to 84 years'] + data['85 years and over']
    # midcareer col
    data['total_midcareer (25-34)'] = data['25 to 34 years']
    data['total_midcareer (35-44)'] = data['35 to 44 years']
    
    data = data.rename(columns={'Total population': 'total_population'})
    data = data.drop(columns=['total_population', '25 to 34 years', '35 to 44 years', '65 to 74 years', '75 to 84 years', '85 years and over'])
    
    
    return data