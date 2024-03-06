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
    df = data
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(df.index[0])
    df = df.reset_index(drop=True)
    df = df.drop(columns = 'Geography')
    df = df.rename(columns = {'Geographic Area Name': 'zip'})
    if year < 2017:
        df = df[['zip', 'Estimate!!SEX AND AGE!!25 to 34 years', 'Estimate!!SEX AND AGE!!35 to 44 years', 'Estimate!!SEX AND AGE!!65 to 74 years', 'Estimate!!SEX AND AGE!!75 to 84 years', 'Estimate!!SEX AND AGE!!85 years and over' 
        ]]
        df['zip'] = df['zip'].map(lambda x: x[6:])
        df = df.apply(pd.to_numeric, errors='ignore')
        df['total_retirement'] = df['Estimate!!SEX AND AGE!!65 to 74 years'] + df['Estimate!!SEX AND AGE!!75 to 84 years'] + df['Estimate!!SEX AND AGE!!85 years and over']
        df = df.rename(columns = {'Estimate!!SEX AND AGE!!25 to 34 years':'total_midcareer (25-34)', 'Estimate!!SEX AND AGE!!35 to 44 years': 'total_midcareer (35-44)'})
        df = df.drop(columns = ['Estimate!!SEX AND AGE!!65 to 74 years', 'Estimate!!SEX AND AGE!!75 to 84 years', 'Estimate!!SEX AND AGE!!85 years and over'])
    else:
        df = df[['zip', 'Estimate!!SEX AND AGE!!Total population!!25 to 34 years', 'Estimate!!SEX AND AGE!!Total population!!35 to 44 years', 'Estimate!!SEX AND AGE!!Total population!!65 to 74 years', 'Estimate!!SEX AND AGE!!Total population!!75 to 84 years', 'Estimate!!SEX AND AGE!!Total population!!85 years and over' 
        ]]
        df['zip'] = df['zip'].map(lambda x: x[6:])
        df = df.apply(pd.to_numeric, errors='ignore')
        df['total_retirement'] = df['Estimate!!SEX AND AGE!!Total population!!65 to 74 years'] + df['Estimate!!SEX AND AGE!!Total population!!75 to 84 years'] + df['Estimate!!SEX AND AGE!!Total population!!85 years and over']
        df = df.rename(columns = {'Estimate!!SEX AND AGE!!Total population!!25 to 34 years':'total_midcareer (25-34)', 'Estimate!!SEX AND AGE!!Total population!!35 to 44 years': 'total_midcareer (35-44)'})
        df = df.drop(columns = ['Estimate!!SEX AND AGE!!Total population!!65 to 74 years', 'Estimate!!SEX AND AGE!!Total population!!75 to 84 years', 'Estimate!!SEX AND AGE!!Total population!!85 years and over'])

    df = df[df['zip'].apply(lambda x: x in params['zip_codes'])]
    df = df.reset_index(drop=True)
    df['year'] = year
    
    return df