import sys
sys.path.insert(0, 'src/helper')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from mlxtend.feature_selection import SequentialFeatureSelector

import tensorflow as tf

from tqdm import tqdm

from feedback import FeedBack
from window_generator import WindowGenerator


MAX_EPOCHS = 50


def train_test_split_by_year(data, end_year):
    data_train = data[data['year'] <= end_year]
    data_test = data[data['year'] > end_year]
    return data_train, data_test

def standardize_data(data_train, data_test):
    train_mean = data_train.mean()
    train_mean['zip'] = 0
    train_std = data_train.std()
    train_std['zip'] = 1
    
    data_train_standardized = (data_train - train_mean) / train_std
    data_test_standardized = (data_test - train_mean) / train_std
    
    return data_train_standardized, data_test_standardized, (train_mean, train_std)

def unstandardize_series(ser, mean, std):
    return (ser*std)+mean

def convert_to_ohe(data_train, data_test):
    
    preproc = ColumnTransformer([('onehots', OneHotEncoder(handle_unknown='ignore'), ['zip'])]
                             ,remainder = 'passthrough')
    data_ohe_train = preproc.fit_transform(data_train)
    
    feature_names = preproc.get_feature_names_out()
    feature_names = np.char.replace(feature_names.astype('str'), 'onehots__','')
    feature_names = np.char.replace(feature_names, 'remainder__','')
    
    data_ohe_train = pd.DataFrame(data_ohe_train, columns=feature_names)
    
    data_ohe_test = preproc.transform(data_test)
    data_ohe_test = pd.DataFrame(data_ohe_test, columns=feature_names)
    
    return data_ohe_train, data_ohe_test

def split_by_zip_code(data_train, data_test, window, ignore_test=False):
    
    data_train_by_zc_tf = {}
    for zip_code in data_train.filter(like='zip').columns:
        data_by_zc = data_train[data_train[zip_code]==1]
        data_train_by_zc_tf[zip_code] = window.make_dataset(data_by_zc)
        
    
    data_test_by_zc_tf = {}
    
    if not ignore_test:
        for zip_code in data_test.filter(like='zip').columns:
            data_by_zc = data_test[data_test[zip_code]==1]
            data_test_by_zc_tf[zip_code] = window.make_dataset(data_by_zc)
        
    return data_train_by_zc_tf, data_test_by_zc_tf

def compile_and_fit(model, data_train_by_zc, data_test_by_zc, num_epochs, train_mean, train_std):
    
    KERAS_VERBOSITY = 0
    patience = 4

    losses = []
    val_losses = []

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.losses.MeanSquaredError()])
    
    for epoch in tqdm(np.arange(num_epochs)):
        
        if (len(losses) >= 2) and (np.abs(losses[-1] - losses[-2]) < 0.1):
            patience -= 1
        if patience <= 0:
            break
        
        loss_curr_epoch = 0
        val_loss_curr_epoch = 0
        i = 0
        
        data_train_by_zip = list(data_train_by_zc.values())
        data_test_by_zip = list(data_test_by_zc.values())
        
        for i in np.arange(len(data_train_by_zip)):
            
            history = model.fit(data_train_by_zip[i], epochs=1, validation_data=data_train_by_zip[i], verbose=KERAS_VERBOSITY)
            loss_curr_epoch += history.history['loss'][0]
            val_loss_curr_epoch += history.history['val_loss'][0]
            i += 1
                
        losses += [np.sqrt(unstandardize_series(loss_curr_epoch/i, train_mean['est'], train_std['est']))]
        val_losses += [np.sqrt(unstandardize_series(val_loss_curr_epoch/i, train_mean['est'], train_std['est']))]
                
    return losses, val_losses

def sum_auto_wide_plot_model(model, wide_data, window, extra_steps, train_mean, train_std):
    
    total_preds = None
    total_inputs = None
    plot_col_index = window.column_indices['est']
    if window.label_columns:
        label_col_index = window.label_columns_indices.get('est', None)
    else:
        label_col_index = plot_col_index
    model.out_steps += extra_steps
    
    for zc in wide_data.keys():
        
        inputs, labels = next(iter(wide_data[zc]))
        
        predictions = model(inputs, training=False)
        curr_preds = unstandardize_series(predictions[0, :, label_col_index], train_mean['est'], train_std['est'])
        curr_inputs = unstandardize_series(inputs[0, :, plot_col_index], train_mean['est'], train_std['est'])
        
        if total_preds is None:
            total_preds = curr_preds
        else:
            total_preds += curr_preds
            
        if total_inputs is None:
            total_inputs = curr_inputs
        else:
            total_inputs += curr_inputs
            
    model.out_steps -= extra_steps
    
    return total_inputs, total_preds

def run(forecast_year=2050):
    """
    Lorem Ipsum
    """

    # Load Data

    data_file_path = 'src/data/temp/zbp_totals_with_features.csv'
    data = pd.read_csv(data_file_path)
    
    # Process Data

    data = data.drop(columns=data.select_dtypes(exclude=['int64', 'float64']).columns)

    end_year = 2020
    short_data_train, short_data_test = train_test_split_by_year(data, end_year)

    end_year = 2018
    long_data_train, long_data_test = train_test_split_by_year(data, end_year)

    short_std_data_train, short_std_data_test, short_train_stats = standardize_data(short_data_train, short_data_test)
    short_train_mean, short_train_std = short_train_stats

    long_std_data_train, long_std_data_test, long_train_stats = standardize_data(long_data_train, long_data_test)
    long_train_mean, long_train_std = long_train_stats

    short_ohe_data_train, short_ohe_data_test = convert_to_ohe(short_std_data_train, short_std_data_test)
    long_ohe_data_train, long_ohe_data_test = convert_to_ohe(long_std_data_train, long_std_data_test)

    # Windows

    OUT_STEPS = 3
    multi_window = WindowGenerator(input_width=OUT_STEPS,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS,
                                train_df=long_ohe_data_train, 
                                test_df=long_ohe_data_test)
    multi_wide_short_data_train_by_zc_tf, multi_wide_short_data_test_by_zc_tf = split_by_zip_code(short_ohe_data_train, short_ohe_data_test, multi_window, ignore_test=True)
    multi_wide_long_data_train_by_zc_tf, multi_wide_long_data_test_by_zc_tf = split_by_zip_code(long_ohe_data_train, long_ohe_data_test, multi_window)

    # FeedBack Model

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
        losses, val_losses = compile_and_fit(short_feedback_model, multi_wide_short_data_train_by_zc_tf, multi_wide_short_data_test_by_zc_tf, MAX_EPOCHS, long_train_mean, long_train_std)
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
        losses, val_losses = compile_and_fit(long_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_wide_long_data_test_by_zc_tf, MAX_EPOCHS)
        long_feedback_model.save_weights(long_feedback_model_filepath)

    # Plot Forecasts
        
    last_pred_year  = forecast_year
    auto_regressive_steps = last_pred_year-2017

    short_feedback_sum_inputs, short_feedback_sum_preds = sum_auto_wide_plot_model(short_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_window, auto_regressive_steps, long_train_mean, long_train_std)
    short_feedback_input_indicies = np.arange(2012, 2012 + short_feedback_sum_inputs.shape[0])
    short_feedback_preds_indicies = np.arange(short_feedback_input_indicies[0] + 3, short_feedback_input_indicies[-1] + auto_regressive_steps + 4)
    fig, ax = plt.subplots()
    plt.plot(short_feedback_input_indicies, short_feedback_sum_inputs, marker='o')
    plt.plot(short_feedback_preds_indicies, short_feedback_sum_preds, marker='X', color='#ff7f0e')
    plt.savefig(f'out/plots/short_feedback_{last_pred_year}_forecasts.jpg', dpi=500, bbox_inches='tight', pad_inches=0)

    long_feedback_sum_inputs, long_feedback_sum_preds = sum_auto_wide_plot_model(long_feedback_model, multi_wide_long_data_train_by_zc_tf, multi_window, auto_regressive_steps, long_train_mean, long_train_std)
    long_feedback_input_indicies = np.arange(2012, 2012 + long_feedback_sum_inputs.shape[0])
    long_feedback_preds_indicies = np.arange(long_feedback_input_indicies[0] + 3, long_feedback_input_indicies[-1] + auto_regressive_steps + 4)
    fig, ax = plt.subplots()
    plt.plot(long_feedback_input_indicies, long_feedback_sum_inputs, marker='o')
    plt.plot(long_feedback_preds_indicies, long_feedback_sum_preds, marker='X', color='#ff7f0e')
    plt.savefig(f'out/plots/long_feedback_{last_pred_year}_forecasts.jpg', dpi=500, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    run()