import numpy as np
import pandas as pd

def reformat_pop(x):
    """
    Reformats and converts to int, string representations of population counts

    Parameters
    ----------
    x: str
        String representation of population count

    Returns
    -------
    str:
        Cleaned integer representation of population count

    Example
    -------
        >>> reformat_pop('1,234')
        1234
    """
    return float(x.replace(',', ''))

# Reformats zip column to be  5 digit zip code
def reformat_zip(x):
    return x[6:11]

def process_pop(data, year, params):
    """
    Processes a single dataset in src/data/raw/pop_age_data.

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
        Dataset with columns ['zip', 'total_population', 'year']
    """
    pop = data.T
    pop.columns = pop.iloc[0]
    pop = pop.drop(pop.index[0])
    pop = pop.reset_index()
    pop = pop.rename_axis(None, axis=1)
    pop = pop.rename(columns = {'index':'zip'})
    #Filtering population data by estimates only
    pop = pop[pop["zip"].str.contains(r'^(?=.*Estimate)')]
    pop['zip'] = pop['zip'].apply(reformat_zip)
    pop_column = pop.rename(columns = {'\xa0\xa0\xa0\xa0Total population': 'Total population'})
    pop['Total population'] = pop_column['Total population'].iloc[:,:1]
    pop = pop[['zip','Total population']]
    pop['year'] = year
    #Converting columns to appropriate datatypes
    pop['zip'] = pop['zip'].astype('int64')
    pop['Total population'] = pop['Total population'].astype(str)
    pop['Total population'] = pop['Total population'].apply(reformat_pop)
    # Keeping only relevant zip-codes
    pop = pop[pop['zip'].apply(lambda x: x in params['zip_codes'])]

    pop = pop.rename(columns={'Total population': 'total_population'})

    return pop