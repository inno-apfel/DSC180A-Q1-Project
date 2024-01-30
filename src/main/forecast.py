import sys
sys.path.insert(0, 'src/helper')

import warnings
warnings.filterwarnings('ignore')

import json

import numpy as np
import pandas as pd

import zbp_visualizer
from arima_forecast import ARIMAForecast

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def run_sklearn_model(model, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates given sklearn model on given data.

    Parameters
    ----------
    model: sklearn.model
        Any sklearn model that can applied on columnar data
    data:
        All X prefixed data below is assumed to contain the columns: ['zip', 'year']
            X_train: pandas.DataFrame
                Data of indendepent variables used to train the model
            y_train: pandas.Series
                Data of dependent variables used to train the model
            X_test: pandas.DataFrame
                Data of indendepent variables used to evaluate the model
            y_test: pandas.Series
                Data of dependent variables used to evaluate the model
    
    Returns
    -------
    tuple:
        pl: sklearn.pipeline.Pipeline
            The trained model
        rmse: float
            The evaluated RMSE for the trained model
        preds_by_zip_year: pandas.DataFrame
            A table containing the predictions the model was evaluated on.
            Contains the columns: ['zip', 'year', 'emp_pred']
    """
    preproc = ColumnTransformer([('onehots', OneHotEncoder(handle_unknown='ignore'), ['zip'])]
                             ,remainder = 'passthrough')
    pl = Pipeline(steps=[('preproc', preproc), ('lr', model)])
    pl.fit(X_train, y_train)
    preds = pl.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    preds_by_zip_year = X_test
    preds_by_zip_year['emp_pred'] = preds
    preds_by_zip_year = preds_by_zip_year[['zip', 'year', 'emp_pred']]
    return (pl, rmse, preds_by_zip_year)

def run_arima_model(year_to_forecast, data_train, data_test):
    """
    Trains and evaluates given sklearn model on given data.

    Parameters
    ----------
    year_to_forecast: int
        The year to forecast up to
    data:
        Assumes all below data includes the columns: ['zip', 'year', 'emp']
            data_train: pandas.DataFrame
                The data to train the model on
            data_test: pandas.DataFrame
                The data to evaluate the model on
    
    Returns
    -------
    tuple:
        model: arima_forecast.ARIMAForecast
            The trained model
        rmse: float
            The evaluated RMSE for the trained model
        preds_by_zip_year: pandas.DataFrame
            A table containing the predictions the model was evaluated on.
            Contains the columns: ['zip', 'year', 'emp_pred']
    """
    data_train['year'] = pd.to_datetime(data_train['year'], format='%Y')
    data_test['year'] = pd.to_datetime(data_test['year'], format='%Y')
    model = ARIMAForecast(data_train, 1, 1, 1)
    model.train()
    preds = model.forecast(year_to_forecast)
    preds_with_true = preds.merge(data_test, on=['zip', 'year'], suffixes=('_pred', '_true'))[['zip', 'year', 'emp_true', 'emp_pred']]
    rmse = mean_squared_error(preds_with_true['emp_true'], preds_with_true['emp_pred'], squared=False)
    preds_by_zip_year = preds_with_true.drop(columns=['emp_true'])
    preds_by_zip_year['year'] = preds_by_zip_year['year'].dt.year

    return (model, rmse, preds_by_zip_year)

def visualize_predictions(preds, year, model_name):
    """
    Generates a chloropleth using prediction data from our forecast models

    Parameters
    ----------
    preds: pd.DataFrame
        A table containing information on a set of predictions
        Assumed to contain the columns: ['zip', 'year', 'emp_pred']
    year: int
        The year to filter by before plotting
    model_name: str
        The name of the model used to generate preds
        Used as a prefix for the naming tag used in generating the visualization
    """
    preds = preds[preds['year'] == year]
    preds = preds[['zip', 'emp_pred']]
    zbp_visualizer.generate_zbp_chloropleth(preds, 'zip', 'emp_pred', f'out/plots/zbp_plot_{model_name}_{year}_preds.html')

def eval_sandag_s14(year_up_to, data):
    """
    Evalutes the SANDAG Series 14 Forecasts for Jobs by ZIP Code against given observations

    Parameters
    ----------
    year_up_to: int
        The last year to evaluate on. Will evaluate on all observations on data up to this year.
    data: pandas.DataFrame
        The true observations to evaluate SANDAG Series 14 Forecasts against.

    Returns
    -------
    float:
        Evaluated RMSE of SANDAG Series 14 Forecasts
    """
    s14_file_path = 'src/data/raw/Series_14_Forecasts/Series_14_Forecasts_Jobs_by_ZIP_Code.csv'
    s14_jobs = pd.read_csv(s14_file_path).rename(columns={'yr_id':'year', 'jobs':'emp'})
    s14_jobs = s14_jobs.groupby(['zip', 'year'])[['emp']].sum().reset_index()
    # our CBP observed data only goes up to 2021
    s14_jobs_pre2021 = s14_jobs[s14_jobs['year'] <= year_up_to]
    sandag_s14_with_trues = data.merge(s14_jobs_pre2021, on=['zip', 'year'], suffixes=('_trues', '_preds'))
    return mean_squared_error(sandag_s14_with_trues['emp_trues'], sandag_s14_with_trues['emp_preds'], squared=False)


def run(end_year):
    """
    Trains and Evaluates three models: [sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor, arima_forecast.ARIMAForecast] 
    on our master ZIP Code County Business Patterns data. 
    And generates chloropleth maps for predictions from the last year forecasted.
    Last forecasted year is determined by last year in the testing data.

    Parameters
    ----------
    end_year: int
        The year to split training data and testing data on. 
        Data for end_year is included in training.
    
    Out
    ---
    zbp_plot_lin_reg_{last_year}_preds: HTML file
        Chloropleth map for last predicted year predictions for our Linear Regression model.
    zbp_plot_random_forest_{last_year}_preds: HTML file
        Chloropleth map for last predicted year predictions for our Random Forest model.
    zbp_plot_arima_{last_year}_preds: HTML file
        Chloropleth map for last predicted year predictions for our ensembled ARIMA model.
    """
    # LOAD FEATURES DATA
    file_path = 'src/data/temp/zbp_totals_with_features.csv'
    data = pd.read_csv(file_path)

    # TRAIN-TEST SPLIT
    data_train = data[data['year'] <= end_year]
    data_test = data[data['year'] > end_year]
    included_feats = ['zip', 'year', 'naics_11_pct', 'naics_21_pct', 'naics_22_pct', 'naics_23_pct',
                    'naics_31_pct', 'naics_42_pct', 'naics_44_pct', 'naics_48_pct',
                    'naics_51_pct', 'naics_52_pct', 'naics_53_pct', 'naics_54_pct',
                    'naics_55_pct', 'naics_56_pct', 'naics_61_pct', 'naics_62_pct',
                    'naics_71_pct', 'naics_72_pct', 'naics_81_pct', 'naics_99_pct',
                    'n1_4_pct', 'n5_9_pct', 'n10_19_pct', 'n20_49_pct', 'n50_99_pct',
                    'n100_249_pct', 'n250_499_pct', 'n500_999_pct', 'n1000_pct']
    X_train = data_train[included_feats]
    y_train = data_train['emp']
    X_test = data_test[included_feats]
    y_test = data_test['emp']

    last_test_year = data_test['year'].max()

    lr_model, lr_rmse, lr_preds = run_sklearn_model(LinearRegression(), X_train, y_train, X_test, y_test)
    rf_model, rf_rmse, rf_preds = run_sklearn_model(RandomForestRegressor(), X_train, y_train, X_test, y_test)
    arima_model, arima_rmse, arima_preds = run_arima_model(last_test_year, data_train, data_test)
    s14_rmse = eval_sandag_s14(last_test_year, data)

    model_comparison_df = pd.DataFrame({'model': ['arima', 'lin-reg', 'random_forest', 'SANDAG-series14'],
                                        'rmse': [arima_rmse, lr_rmse, rf_rmse, s14_rmse]})

    print()
    print(model_comparison_df.to_string(index=False))

    visualize_predictions(lr_preds, last_test_year, 'lin_reg')
    visualize_predictions(rf_preds, last_test_year, 'random_forest')
    visualize_predictions(arima_preds, last_test_year, 'arima')


if __name__ == '__main__':
    with open('../../config/config.json', 'r') as fh:
        params = json.load(fh)
    print()
    print('-----------------------------------------------------------')
    print(f"Models will be trained on data from {min(params['years'])} to <input-year>")
    print(f"Models will be evaluated on data from <input-year> to {max(params['years'])}")
    print(f"Current year range for available data: {min(params['years'])} - {max(params['years'])}")
    print('-----------------------------------------------------------')
    end_year = int(input('(input-year): '))
    run(end_year)
    print()
    print('View model prediction visualizations in out/plots')
    print()