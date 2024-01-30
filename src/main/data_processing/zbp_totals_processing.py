import numpy as np
import pandas as pd

def process_zbp_totals(data, year, params):
    """
    Processes a single ZBP Totals dataset in src/data/raw/zbp_data/zbptotals

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
        Dataset with columns ['zip', 'emp_nf', 'emp', 'qp1_nf', 'qp_1', 'ap_nf', 'ap', 'est', 'year']
    """
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