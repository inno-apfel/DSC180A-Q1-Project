import numpy as np
import pandas as pd


def process_retire_data(data, year, params):
    
    column_names = data.columns
    # Only getting Estimates
    filtered_columns = [col for col in column_names if col[-8:] == 'Estimate' and col[6:11].isdigit()]

    data = data[filtered_columns]

    # Cleaning up so that the rows are just the zipcode
    data.columns = data.columns.str.replace('ZCTA5 ', '').str.replace('!!Estimate', '')
    
    # Filter columns based on San Diego County zip codes

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
    data = data.applymap(lambda x: str(x).replace(',', ''))

    # Convert all columns to numeric type
    data = data.apply(pd.to_numeric, errors='ignore')

    # Creating row that is total retirement population
    new_row = data.iloc[3] + data .iloc[4] + data.iloc[5]

    new_row.name = 'Total Retirement'

    # Append the new row to the DataFrame
    data = data.append(new_row)
    
    # Creating row that is total mid career population
    new_row = data.iloc[1] + data .iloc[2]

    new_row.name = 'Total Mid Career'
    
    # Append the new row to the DataFrame
    data = data.append(new_row)
    
    data = data.loc[['Total population', 'Total Retirement', 'Total Mid Career']]
    
    data = data.reset_index().T

    # Assign year variable
    data = data.assign(year=np.full(data.shape[0], year))
    
    data = data.drop(data.index[0])
    
    data = data.reset_index()
    
    data = data.rename(columns={0: 'total_population', 1: 'total_retirement', 2: 'total_mid_career', 'index': 'zip'})

    data = data.drop(columns=['total_population'])

    data['zip'] = data['zip'].astype('int64').apply(lambda x: x in params['zip_codes'])
    
    return data