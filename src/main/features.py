import numpy as np
import pandas as pd

def run():
    """
    Produces the necessary features used for our forecasting models

    Out
    ---
    zbp_totals_with_features: CSV file
        A master dataset indexed by ZIP Code, containing information from ZBP Details and Totals datasets
        alongside the features: [naics_x_pct, ni_j_pct]
    """
    # LOAD PROCESSED DATA
    zbp_details_file_path = 'src/data/temp/processed_zbp_detail_data.csv'
    zbp_detail = pd.read_csv(zbp_details_file_path)

    zbp_totals_file_path = 'src/data/temp/processed_zbp_totals_data.csv'
    zbp_totals = pd.read_csv(zbp_totals_file_path)

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
    zbp_totals_with_features.to_csv('src/data/temp/zbp_totals_with_features.csv', index=False)


if __name__ == '__main__':
    run()