import sys
import json

import pandas as pd

sys.path.insert(0, 'src/main/data_processing')
import zbp_detail_processing
import zbp_totals_processing
import household_income_processing
import population_counts_processing
import retirement_processing
import household_counts_processing
import unemployment_processing


def run(params):
    """
    Processes all datasets in the src/data/raw/ directory, saving their processed states in src/data/temp/.

    Out
    -----
    processed_zbp_detail_data: CSV file
        Combined processed data for all ZIP Code Industry Details data 
    processed_zbp_totals_data: CSV file
        Combined processed data for all ZIP Code Totals data 
    processed_hh_income_data: CSV file
        Dataset containing median household income data by (zip, year) pairs
    processed_total_pop_data: CSV file
        Dataset containing total population count data by (zip, year) pairs
    processed_retire_detail_data: CSV file
        Dataset containing population count data binned by age demographics by (zip, year) pairs
    processed_household_counts_data: CSV file
        Dataset containing total household counts data by (zip, year) pairs
    processed_unemployment_data: CSV file
        Dataset containing unemployment rates by year for the San Diego County
    """
    
    # LOAD DATA
    zbp_detail_by_year = {}
    zbp_totals_by_year = {}
    hh_income_by_year = {}
    total_pop_by_year = {}
    retire_by_year = {}
    household_counts_by_year = {}
    for year in params['years']:
        shortened_year = str(year)[2:]
        detail_encoding = None
        totals_encoding = None
        if year >= 2017:
            detail_encoding = 'latin-1'
        if year >= 2018:
            totals_encoding = 'latin-1'
        zbp_detail_by_year[year] = pd.read_csv(f'src/data/raw/zbp_data/zbpdetail/zbp{shortened_year}detail/zbp{shortened_year}detail.txt', encoding=detail_encoding)
        zbp_totals_by_year[year] = pd.read_csv(f'src/data/raw/zbp_data/zbptotals/zbp{shortened_year}totals/zbp{shortened_year}totals.txt', encoding=totals_encoding)
        hh_income_by_year[year] = pd.read_csv(f'src/data/raw/household_data/ACSST5Y{year}.csv')
        total_pop_by_year[year] = pd.read_csv(f'src/data/raw/pop_age_data/ACSDP5Y{year}.csv')
        retire_by_year[year] = pd.read_csv(f'src/data/raw/retire_data/ACSDP5Y{year}.DP05-Data.csv')
        household_counts_by_year[year] = pd.read_csv(f'src/data/raw/acs_data/household_counts_data/household_counts_{year}.csv', skiprows=1)
    unemployment_by_year = pd.read_csv('src/data/raw/employment_data/employment.csv')

    # PROCESS AND CONCATENATE DATA
    for year in zbp_detail_by_year:
        zbp_detail_by_year[year] = zbp_detail_processing.process_zbp_data(zbp_detail_by_year[year], year, params)

    for year in zbp_totals_by_year:
        zbp_totals_by_year[year] = zbp_totals_processing.process_zbp_totals(zbp_totals_by_year[year], year, params)

    for year in hh_income_by_year:
        hh_income_by_year[year] = household_income_processing.process_hh(hh_income_by_year[year], year, params)

    for year in total_pop_by_year:
        total_pop_by_year[year] = population_counts_processing.process_pop(total_pop_by_year[year], year, params)

    for year in retire_by_year:
        retire_by_year[year] = retirement_processing.process_retire_data(retire_by_year[year], year, params)

    for year in household_counts_by_year:
        household_counts_by_year[year] = household_counts_processing.process_household_counts_data(household_counts_by_year[year], year, params)

    unemployment_by_year = unemployment_processing.process_unemployment_data(unemployment_by_year)
    
    # SAVE PROCESSED DATA
    zbp_detail_data = pd.concat(list(zbp_detail_by_year.values()), ignore_index=True).reset_index(drop=True)
    zbp_detail_data.to_csv('src/data/temp/processed_zbp_detail_data.csv', index=False)

    zbp_totals_data = pd.concat(list(zbp_totals_by_year.values()), ignore_index=True).reset_index(drop=True)
    zbp_totals_data.to_csv('src/data/temp/processed_zbp_totals_data.csv', index=False)

    hh_income_data = pd.concat(list(hh_income_by_year.values()), ignore_index=True).reset_index(drop=True)
    hh_income_data.to_csv('src/data/temp/processed_hh_income_data.csv', index=False)

    total_pop_data = pd.concat(list(total_pop_by_year.values()), ignore_index=True).reset_index(drop=True)
    total_pop_data.to_csv('src/data/temp/processed_total_pop_data.csv', index=False)

    retire_data = pd.concat(list(retire_by_year.values()), ignore_index=True).reset_index(drop=True)
    retire_data.to_csv('src/data/temp/processed_retire_detail_data.csv', index=False)

    household_counts_data = pd.concat(list(household_counts_by_year.values()), ignore_index=True).reset_index(drop=True)
    household_counts_data.to_csv('src/data/temp/processed_household_counts_data.csv', index=False)

    unemployment_by_year.to_csv('src/data/temp/processed_unemployment_data.csv', index=False)


if __name__ == '__main__':
    with open('config/config.json', 'r') as fh:
        params = json.load(fh)
    run(params)