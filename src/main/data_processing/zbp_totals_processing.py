import numpy as np
import pandas as pd

def process_zbp_totals(data, year, params):
    # drop naming columns
    cols = ['name', 'city', 'stabbr', 'cty_name']
    if year <= 2017:
        cols += ['empflag']
    data = data.drop(columns=cols)
    # filter only relavent zip codes
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    # assign year variable
    data = data.assign(year=np.full(data.shape[0], year))
    return data