import numpy as np
import pandas as pd

#Removing commas from population strings before converting to floats.
def reformat_pop(x):
    return float(x.replace(',', ''))

# Reformats zip column to be  5 digit zip code
def reformat_zip(x):
    return x[6:11]

def float_check(x):
    return isinstance(x, float)

#Function for processing population data by year
def process_pop(data, year, params):
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
    return pop