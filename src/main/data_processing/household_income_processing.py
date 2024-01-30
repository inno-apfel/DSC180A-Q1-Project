import sys
sys.path.insert(0, 'src/helper')

import numpy as np
import pandas as pd

from data_processing_helpers import reformat_zip
from data_processing_helpers import reformat_income

def process_hh(data, year, params):
    """
    Processes a single dataset in src/data/raw/household_data.

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
        Dataset with columns ['zip', 'median_hh_income', 'year']
    """
    hh2012 = data.T
    hh2012.columns = hh2012.iloc[0]
    hh2012 = hh2012.drop(hh2012.index[0])
    hh2012 = hh2012.reset_index()
    hh2012 = hh2012.rename_axis(None, axis=1)
    hh2012 = hh2012.rename(columns = {'index':'zip'})
    # Filtering data by household estimates
    hh2012 = hh2012[hh2012["zip"].str.contains(r'^(?=.*Households)(?=.*Estimate)')]
    hh2012['zip'] = hh2012['zip'].apply(reformat_zip)
    hh2012 = hh2012[['zip','Median income (dollars)']]
    hh2012['year'] = year
    # Dropping missing values
    hh2012 = hh2012[hh2012['Median income (dollars)'].str.contains('-')==False]
    hh2012 = hh2012[hh2012['Median income (dollars)'].str.contains('X')==False]
    hh2012['Median income (dollars)'] = hh2012['Median income (dollars)'].apply(reformat_income)
    hh2012['zip'] = hh2012['zip'].astype('int64')
    # Keeping only relevant zip-codes
    hh2012 = hh2012[hh2012['zip'].apply(lambda x: x in params['zip_codes'])]
    hh2012 = hh2012.rename(columns={'Median income (dollars)': 'median_hh_income'})
    return hh2012