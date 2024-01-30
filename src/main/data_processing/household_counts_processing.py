import sys
sys.path.insert(0, 'src/helper')

import numpy as np
import pandas as pd

from data_processing_helpers import reformat_zip

def process_household_counts_data(data, year, params):
    """
    Processes a single dataset in src/data/raw/acs_data/household_counts_data.

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
        Dataset with columns ['zip', 'total_households', 'year']
    """
    data = data[['Geographic Area Name', 'Estimate!!HOUSEHOLDS BY TYPE!!Total households']]
    data = data.rename(columns={'Geographic Area Name': 'zip',
                                'Estimate!!HOUSEHOLDS BY TYPE!!Total households': 'total_households'})
    data['zip'] = data['zip'].apply(reformat_zip).astype('int64')
    data['year'] = year
    data = data[data['zip'].apply(lambda x: x in params['zip_codes'])]
    return data