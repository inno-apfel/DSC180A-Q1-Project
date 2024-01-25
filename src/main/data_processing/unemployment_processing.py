import numpy as np
import pandas as pd

# Reformats zip column to be  5 digit zip code
def reformat_zip(x):
    return x[6:11]

def process_unemployment_data(data, year, params):
    data = data[['Geographic Area Name', 'Estimate!!HOUSEHOLDS BY TYPE!!Total households']]
    data = data.rename(columns={'Geographic Area Name': 'zip',
                                'Estimate!!HOUSEHOLDS BY TYPE!!Total households': 'total_households'})
    data['zip'] = data['zip'].apply(reformat_zip).astype('int64')
    data['year'] = year
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    return data