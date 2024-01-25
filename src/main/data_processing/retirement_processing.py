import numpy as np
import pandas as pd

def process_retire_data(data, year, params):
    
    column_names = data.columns
    # Only getting Estimates
    filtered_columns = [col for col in column_names if col[-8:] == 'Estimate' and col[6:11].isdigit()]
    data = data[filtered_columns]
    # Cleaning up so that the rows are just the zipcode
    data.columns = data.columns.str.replace('ZCTA5 ', '').str.replace('!!Estimate', '')
    # Selecting: Total population and retired aged
    selected_rows = data.loc[[1, 14, 15, 16]]
    # Define the new row names
    og_names = {
                    1: 'Total population',
                    14: '65 to 74 years',
                    15: '75 to 84 years',
                    16: '85 years and over'
                }
    data = selected_rows.rename(index=og_names)
    # Remove commas from all numbers in all columns
    data = data.applymap(lambda x: str(x).replace(',', ''))
    # Convert all columns to numeric type
    data = data.apply(pd.to_numeric, errors='ignore')
    # Transpose dataframe and convert zip-code col
    data = data.T
    data.index = data.index.set_names(['zip'])
    data = data.reset_index()
    data['zip'] = data['zip'].astype('int64')
    # Creating row for total retirement population
    data.insert(2, 'Total Retirement', data.iloc[:,2:].sum(axis=1))
    # Assign year variable
    data = data.assign(year=np.full(data.shape[0], year))
    # Keeping only relevant zip-codes
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    data = data.rename({'Total population': 'total_population',
                        'Total Retirement': 'total_retirement',
                         '65 to 74 years': 'population_between_65_to_74',
                         '75 to 84 years': 'population_between_75_to_84',
                         '85 years and over': 'population_85_and_older'})
    data = data.drop(columns=['total_population'])
    
    return data