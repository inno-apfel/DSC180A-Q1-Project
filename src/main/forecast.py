import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, 'src/helper')
import model_helpers
from feedback import FeedBack
from window_generator import WindowGenerator
from zbp_visualizer import generate_zbp_chloropleth
from arima_forecast import ARIMAForecast


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

MAX_EPOCHS = 50

def run(forecast_year=2050):
    """
    Generates chloropleth visualizations of Establishment counts for the last year in test set for every non-tensorflow model.
    Trains an autoregressive feedback LSTM and generates aggregated region level lineplot for forecasts up to a given year.

    Parameters
    ----------
    forecast_year: int
        year to forecast up to for feedback LSTM visualization
    """

    # LOAD DATA

    data_file_path = 'src/data/temp/zbp_totals_with_features.csv'
    data = pd.read_csv(data_file_path)
    lagged_data_file_path = 'src/data/temp/lagged_zbp_totals_with_features.csv'
    lagged_data = pd.read_csv(lagged_data_file_path)
    
    # PROCESS DATA

    data = data.drop(columns=data.select_dtypes(exclude=['int64', 'float64']).columns)
    lagged_data = lagged_data.drop(columns=lagged_data.select_dtypes(exclude=['int64', 'float64']).columns)

    end_year = 2020
    short_data_train, short_data_test = model_helpers.train_test_split_by_year(data, end_year)
    short_lagged_data_train, short_lagged_data_test = model_helpers.train_test_split_by_year(lagged_data, end_year)

    end_year = 2018
    long_data_train, long_data_test = model_helpers.train_test_split_by_year(data, end_year)
    long_lagged_data_train, long_lagged_data_test = model_helpers.train_test_split_by_year(lagged_data, end_year)

    short_std_data_train, short_std_data_test, short_train_stats = model_helpers.standardize_data(short_data_train, short_data_test)
    short_train_mean, short_train_std = short_train_stats

    short_lagged_std_data_train, short_lagged_std_data_test, short_lagged_train_stats = model_helpers.standardize_data(short_lagged_data_train, short_lagged_data_test)
    short_lagged_train_mean, short_lagged_train_std = short_lagged_train_stats

    long_lagged_std_data_train, long_lagged_std_data_test, long_lagged_train_stats = model_helpers.standardize_data(long_lagged_data_train, long_lagged_data_test)
    long_lagged_train_mean, long_lagged_train_std = long_lagged_train_stats

    long_std_data_train, long_std_data_test, long_train_stats = model_helpers.standardize_data(long_data_train, long_data_test)
    long_train_mean, long_train_std = long_train_stats

    short_ohe_data_train, short_ohe_data_test = model_helpers.convert_to_ohe(short_std_data_train, short_std_data_test)
    long_ohe_data_train, long_ohe_data_test = model_helpers.convert_to_ohe(long_std_data_train, long_std_data_test)

    short_lagged_ohe_data_train, short_lagged_ohe_data_test = model_helpers.convert_to_ohe(short_lagged_std_data_train, short_lagged_std_data_test)
    long_lagged_ohe_data_train, long_lagged_ohe_data_test = model_helpers.convert_to_ohe(long_lagged_std_data_train, long_lagged_std_data_test)

    # GENERATE ZBP PLOTS FOR MODELS

    def load_predict_plot(model_name, modelpath, data_test, non_ohe_data_test, train_mean, train_std):
        if os.path.isfile(modelpath):
            with open(modelpath,'rb') as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError("    FileNotFoundError: Make sure to run 'models' before 'forecast'")
        
        if isinstance(model, ARIMAForecast):
            final_preds = model.forecast(data_test['year'].max())
        else:
            X_test = data_test.drop(columns=['est'])
            y_test = data_test['est']
            non_ohe_X_test = non_ohe_data_test.drop(columns=['est'])
            non_ohe_X_test['est'] = model.predict(X_test)
            final_preds = model_helpers.unstandardize_series(non_ohe_X_test, train_mean, train_std)[['zip', 'year', 'est']]

        final_preds.to_csv(f'out/forecast_tables/{model_name}_zcta_forecast.csv', index=False)
        final_year = final_preds['year'].unique().max()
        final_preds = final_preds[final_preds['year']==final_year]
        generate_zbp_chloropleth(final_preds, 'zip', 'est', f'out/plots/{model_name}_{int(final_year)}_zcta_forecast.html')

    load_predict_plot('lin_reg', 'out/models/short_lr.pkl', 
                      short_lagged_ohe_data_test, short_lagged_std_data_test,
                      short_lagged_train_mean, short_lagged_train_std)
    
    load_predict_plot('lasso', 'out/models/short_rf.pkl', 
                      short_lagged_ohe_data_test, short_lagged_std_data_test,
                      short_lagged_train_mean, short_lagged_train_std)
    
    load_predict_plot('random_forest', 'out/models/short_lasso.pkl', 
                      short_lagged_ohe_data_test, short_lagged_std_data_test,
                      short_lagged_train_mean, short_lagged_train_std)
    
    load_predict_plot('arima', 'out/models/short_arima.pkl', 
                      short_data_test, None,
                      None, None)

    # CREATE WINDOW OBJECTS FOR TENSORFLOW MODELS
    # CONVERT DATASETS TO TF.DATASETS USING SAID WINDOWS

    OUT_STEPS = 3
    multi_window = WindowGenerator(input_width=OUT_STEPS,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS,
                                train_df=long_ohe_data_train, 
                                test_df=long_ohe_data_test)
    multi_wide_short_data_train_by_zc_tf, multi_wide_short_data_test_by_zc_tf = model_helpers.split_by_zip_code(short_ohe_data_train, short_ohe_data_test, multi_window, ignore_test=True)
    multi_wide_long_data_train_by_zc_tf, multi_wide_long_data_test_by_zc_tf = model_helpers.split_by_zip_code(long_ohe_data_train, long_ohe_data_test, multi_window)

    # Autoreggressive FeedBack LSTM Model

    short_feedback_model_filepath = 'out/models/short_feedback_model_weights.tf'
    try:
        short_feedback_model = FeedBack(units=256, out_steps=OUT_STEPS, num_features=long_ohe_data_train.shape[1])
        short_feedback_model.built = True
        short_feedback_model.load_weights(short_feedback_model_filepath).expect_partial()
        short_feedback_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     metrics=[tf.keras.losses.MeanSquaredError()])
    except:
        short_feedback_model = FeedBack(units=256, out_steps=OUT_STEPS, num_features=long_ohe_data_train.shape[1])
        losses, val_losses = model_helpers.compile_and_fit(short_feedback_model, multi_wide_short_data_train_by_zc_tf, multi_wide_short_data_test_by_zc_tf, MAX_EPOCHS, long_train_mean, long_train_std)
        short_feedback_model.save_weights(short_feedback_model_filepath)

    long_feedback_model_filepath = 'out/models/long_feedback_model_weights.tf'
    try:
        long_feedback_model = FeedBack(units=256, out_steps=OUT_STEPS, num_features=long_ohe_data_train.shape[1])
        long_feedback_model.built = True
        long_feedback_model.load_weights(long_feedback_model_filepath).expect_partial()
        long_feedback_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                    optimizer=tf.keras.optimizers.Adam(),
                                    metrics=[tf.keras.losses.MeanSquaredError()])
    except:
        long_feedback_model = FeedBack(units=256, out_steps=OUT_STEPS, num_features=long_ohe_data_train.shape[1])
        losses, val_losses = model_helpers.compile_and_fit(long_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_wide_long_data_test_by_zc_tf, MAX_EPOCHS)
        long_feedback_model.save_weights(long_feedback_model_filepath)

    # GENERATE REGION LEVEL VISUALIZATIONS FOR LSTM
        
    last_pred_year  = forecast_year
    auto_regressive_steps = last_pred_year-2017

    short_feedback_sum_inputs, short_feedback_sum_preds = model_helpers.sum_auto_wide_plot_model(short_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_window, auto_regressive_steps, long_train_mean, long_train_std)
    short_feedback_input_indicies = np.arange(2012, 2012 + short_feedback_sum_inputs.shape[0])
    short_feedback_preds_indicies = np.arange(short_feedback_input_indicies[0] + 3, short_feedback_input_indicies[-1] + auto_regressive_steps + 4)
    fig, ax = plt.subplots()
    plt.plot(short_feedback_input_indicies, short_feedback_sum_inputs, marker='o')
    plt.plot(short_feedback_preds_indicies, short_feedback_sum_preds, marker='X', color='#ff7f0e')
    plt.savefig(f'out/plots/short_feedback_{last_pred_year}_forecasts.jpg', dpi=500, bbox_inches='tight', pad_inches=0)

    long_feedback_sum_inputs, long_feedback_sum_preds = model_helpers.sum_auto_wide_plot_model(long_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_window, auto_regressive_steps, long_train_mean, long_train_std)
    long_feedback_input_indicies = np.arange(2012, 2012 + long_feedback_sum_inputs.shape[0])
    long_feedback_preds_indicies = np.arange(long_feedback_input_indicies[0] + 3, long_feedback_input_indicies[-1] + auto_regressive_steps + 4)
    fig, ax = plt.subplots()
    plt.plot(long_feedback_input_indicies, long_feedback_sum_inputs, marker='o')
    plt.plot(long_feedback_preds_indicies, long_feedback_sum_preds, marker='X', color='#ff7f0e')
    plt.savefig(f'out/plots/long_feedback_{last_pred_year}_forecasts.jpg', dpi=500, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    run()