import pandas as pd

def run():
    """
    Runs necessary transformation on our processed data to produce the features and final dataset used by our forecasting models

    Out
    ---
    zbp_totals_with_features: CSV file
        A master dataset indexed by ZIP Code, containing information from ZBP Details and Totals datasets
        alongside the features: [naics_x_pct, ni_j_pct]
    lagged_zbp_totals_with_features: CSV file
        zbp_totals_with_features where all features except ['zip', 'year', 'est'] are lagged by one year
    """
    # LOAD PROCESSED DATA
    zbp_details_file_path = 'src/data/temp/processed_zbp_detail_data.csv'
    zbp_detail = pd.read_csv(zbp_details_file_path)

    zbp_totals_file_path = 'src/data/temp/processed_zbp_totals_data.csv'
    zbp_totals = pd.read_csv(zbp_totals_file_path)

    hh_income_file_path = 'src/data/temp/processed_hh_income_data.csv'
    hh_income_data = pd.read_csv(hh_income_file_path)

    total_pop_file_path = 'src/data/temp/processed_total_pop_data.csv'
    total_pop_data = pd.read_csv(total_pop_file_path)

    retire_file_path = 'src/data/temp/processed_retire_detail_data.csv'
    retire_data = pd.read_csv(retire_file_path)

    # PRODUCE FEATURES FOR ZIP-CODE LEVEL INDUSTRY DATA
    # naics_x_pct = num of establishments from industry x in zipcode / total num of establishments in zipcode
    zip_est = zbp_detail.groupby(['zip', 'year'])['est'].sum()
    zip_naics = zbp_detail.groupby(['zip', 'year', 'naics'])['est'].sum()
    naics_percentages = (zip_naics/zip_est).unstack().fillna(value=0).add_prefix('naics_').add_suffix('_pct').reset_index()

    # PRODUCE FEATURES FOR ZIP-CODE LEVEL ESTABLISHMENT SIZE DATA
    # ni_j_pct = num of establishments with between i and j employees in zipcode / total num of establishments in zipcode
    est_size_percentages = zbp_detail.groupby(['zip', 'year'])[['n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']].sum()
    est_size_percentages = est_size_percentages.apply(lambda ser: ser/zip_est, axis=0).add_suffix('_pct').reset_index()

    # MERGE CREATE FEATURES WITH MAIN DATASET (ZBP_TOTALS)
    zbp_totals_with_features = zbp_totals.merge(naics_percentages, on=['zip', 'year'])
    zbp_totals_with_features = zbp_totals_with_features.merge(est_size_percentages, on=['zip', 'year'])
    zbp_totals_with_features = zbp_totals_with_features.merge(hh_income_data, on=['zip', 'year'])
    zbp_totals_with_features = zbp_totals_with_features.merge(total_pop_data, on=['zip', 'year'])
    zbp_totals_with_features = zbp_totals_with_features.merge(retire_data, on=['zip', 'year'])

    # drop zip codes with insufficient observations (<max)
    zip_code_by_num_observations = zbp_totals_with_features.groupby('zip')['year'].count().sort_values()
    zip_codes_with_insufficient_observations = zip_code_by_num_observations[zip_code_by_num_observations<zip_code_by_num_observations.max()].index
    zbp_totals_with_features = zbp_totals_with_features[zbp_totals_with_features['zip'].apply(lambda x: x not in zip_codes_with_insufficient_observations)]

    zbp_totals_with_features.to_csv('src/data/temp/zbp_totals_with_features.csv', index=False)

    # create lagged dataset
    def lag_all_zip_codes(data, cols_not_to_lag, cols_to_lag_and_keep = None, years_to_lag=1):

        def create_lagged_dataset(data, cols_not_to_lag, cols_to_lag_and_keep, years_to_lag):
            cols_to_lag = data.columns.drop(cols_not_to_lag)
            if cols_to_lag_and_keep is not None:
                cols_to_lag = cols_to_lag.append(pd.Index(cols_to_lag_and_keep))
            lagged_data = data[cols_to_lag].shift(years_to_lag)
            if cols_to_lag_and_keep is not None:
                lagged_data = lagged_data.rename(columns={col:col+f'_lag_{years_to_lag}' for col in cols_to_lag_and_keep})
            return data[cols_not_to_lag].join(lagged_data, how='inner')
        
        temp = []
        
        for curr_zip in data['zip'].unique():
            
            curr_zip_data = data[data['zip']==curr_zip].sort_values('year')
            temp += [create_lagged_dataset(curr_zip_data, cols_not_to_lag, cols_to_lag_and_keep=cols_to_lag_and_keep, years_to_lag=years_to_lag).iloc[years_to_lag:,:]]
            
        return pd.concat(temp, ignore_index=True).reset_index(drop=True)
    
    data = pd.read_csv('src/data/temp/zbp_totals_with_features.csv')
    lagged = lag_all_zip_codes(data, ['zip', 'year', 'est'], ['est'])
    lagged.to_csv('src/data/temp/lagged_zbp_totals_with_features.csv', index=False)

    # create dataset for change in # est since last year
    lagged['change_in_est'] = lagged['est'] - lagged['est_lag_1']
    lagged = lagged.drop(columns=['est'])
    lagged.to_csv('src/data/temp/change_in_est_zbp_totals_with_features.csv', index=False)


if __name__ == '__main__':
    run()