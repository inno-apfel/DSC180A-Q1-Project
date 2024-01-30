import numpy as np
import pandas as pd

def process_unemployment_data(data):
    """
    Processes src/data/raw/employment_data/employment.csv

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to processes

    Returns
    -------
    pandas.DataFrame:
        Dataset with columns ['year', 'unemployment_rate']
    """

    data['year'] = pd.to_datetime(data['DATE']).dt.year
    data = data.drop(columns=['DATE','CASAND5URN_20240104'])
    data.rename(columns={'CASAND5URN_20231130': 'unemployment_rate'}, inplace=True)
    unemployment_by_year = data.groupby('year')['unemployment_rate'].mean().reset_index()

    return unemployment_by_year