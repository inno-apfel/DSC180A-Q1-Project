import sys
sys.path.insert(0, 'src/helper')

import os.path
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

from mlxtend.feature_selection import SequentialFeatureSelector

import tensorflow as tf

from tqdm import tqdm

from arima_forecast import ARIMAForecast
from window_generator import WindowGenerator


SEED = 42
MAX_EPOCHS = 50

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        selected_features = [col for col in X.columns if any(name in col for name in self.feature_names)]
        return X[selected_features]


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





def fit_eval(model, data_train, data_test, included_feats, train_mean, train_std, fitted=False):
    
    if included_feats == 'all':
        included_feats = data_train.columns.drop(['est'])
    
    X_train = data_train[included_feats]
    y_train = data_train['est']
    X_test = data_test[included_feats]
    y_test = data_test['est']
    
    if fitted == False:
        model.fit(X_train, y_train)

    y_preds = model.predict(X_train)
    inverted_y_train = unstandardize_series(y_train, train_mean['est'], train_std['est'])
    inverted_y_preds = unstandardize_series(y_preds, train_mean['est'], train_std['est'])
    train_rmse = mean_squared_error(inverted_y_train, inverted_y_preds, squared=False)
    
    y_preds = model.predict(X_test)
    inverted_y_test = unstandardize_series(y_test, train_mean['est'], train_std['est'])
    inverted_y_preds = unstandardize_series(y_preds, train_mean['est'], train_std['est'])
    test_rmse = mean_squared_error(inverted_y_test, inverted_y_preds, squared=False)
    
    return model, train_rmse, test_rmse

def run_grid_search(data_train, data_test, included_feats, model, param_grid):
    
    if included_feats == 'all':
        included_feats = data_train.columns.drop(['est'])

    X_train = data_train[included_feats]
    y_train = data_train['est']
    X_test = data_test[included_feats]
    y_test = data_test['est']
    
    grid_search = GridSearchCV(model, param_grid, scoring = 'neg_root_mean_squared_error', n_jobs = -1)
    grid_search.fit(X_train, y_train)
    
    return grid_search


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

def evaluate_on_all_zip(model, data_train_by_zc, train_mean, train_std):
    total = 0
    i = 0
    for zc in data_train_by_zc.keys():
        total += unstandardize_series(model.evaluate(data_train_by_zc[zc], verbose=0)[0], 
                                      train_mean['est'], train_std['est'])
        i += 1
    return np.sqrt(total/i)

def gen_sum_preds(model, data_test, train_mean, train_std):
    
    total_preds = None
    for zc in data_test.keys():
        curr_preds = model.predict(data_test[zc], verbose=0)[:, 0, 0]
        curr_preds = unstandardize_series(curr_preds, train_mean['est'], train_std['est'])
        if total_preds is None:
            total_preds = curr_preds
        else:
            total_preds += curr_preds
            
    return total_preds



def run():
    """
    Lorem Ipsum
    """

    # Load Data

    data_file_path = 'src/data/temp/zbp_totals_with_features.csv'
    data = pd.read_csv(data_file_path)
    lagged_data_file_path = 'src/data/temp/lagged_zbp_totals_with_features.csv'
    lagged_data = pd.read_csv(lagged_data_file_path)

    # Drop Categorical Features (noise flags)

    data = data.drop(columns=data.select_dtypes(exclude=['int64', 'float64']).columns)
    lagged_data = lagged_data.drop(columns=lagged_data.select_dtypes(exclude=['int64', 'float64']).columns)

    # Train-Test Split

    end_year = 2020
    short_data_train, short_data_test = train_test_split_by_year(data, end_year)
    short_lagged_data_train, short_lagged_data_test = train_test_split_by_year(lagged_data, end_year)

    end_year = 2018
    long_data_train, long_data_test = train_test_split_by_year(data, end_year)
    long_lagged_data_train, long_lagged_data_test = train_test_split_by_year(lagged_data, end_year)

    # Standardize Data

    short_std_data_train, short_std_data_test, short_train_stats = standardize_data(short_data_train, short_data_test)
    short_train_mean, short_train_std = short_train_stats

    long_std_data_train, long_std_data_test, long_train_stats = standardize_data(long_data_train, long_data_test)
    long_train_mean, long_train_std = long_train_stats

    short_lagged_std_data_train, short_lagged_std_data_test, short_lagged_train_stats = standardize_data(short_lagged_data_train, short_lagged_data_test)
    short_lagged_train_mean, short_lagged_train_std = short_lagged_train_stats

    long_lagged_std_data_train, long_lagged_std_data_test, long_lagged_train_stats = standardize_data(long_lagged_data_train, long_lagged_data_test)
    long_lagged_train_mean, long_lagged_train_std = long_lagged_train_stats

    # Ohe Data

    short_ohe_data_train, short_ohe_data_test = convert_to_ohe(short_std_data_train, short_std_data_test)
    long_ohe_data_train, long_ohe_data_test = convert_to_ohe(long_std_data_train, long_std_data_test)

    short_lagged_ohe_data_train, short_lagged_ohe_data_test = convert_to_ohe(short_lagged_std_data_train, short_lagged_std_data_test)
    long_lagged_ohe_data_train, long_lagged_ohe_data_test = convert_to_ohe(long_lagged_std_data_train, long_lagged_std_data_test)

    # Create Tensorflow Test Sets

    last_short_data_year = short_ohe_data_train['year'].unique().max()
    tf_short_ohe_data_train = short_ohe_data_train[short_ohe_data_train['year']<last_short_data_year]
    tf_short_ohe_data_test = pd.concat([short_ohe_data_train[short_ohe_data_train['year']==last_short_data_year], short_ohe_data_test])

    last_long_data_year = long_ohe_data_train['year'].unique().max()
    tf_long_ohe_data_train = long_ohe_data_train[long_ohe_data_train['year']<last_long_data_year]
    tf_long_ohe_data_test = pd.concat([long_ohe_data_train[long_ohe_data_train['year']==last_long_data_year], long_ohe_data_test])

    # TRAIN MODELS

    # Random Forest

    param_grid = {'n_estimators': [50], 
                  'max_depth': [50]}
    
    short_rf_filepath = 'out/models/short_rf.pkl'
    short_rf_fitted = False
    if os.path.isfile(short_rf_filepath):
        with open(short_rf_filepath,'rb') as f:
            short_rf = pickle.load(f)
        short_rf_fitted = True
    else:
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        gs_results = run_grid_search(short_lagged_ohe_data_train, short_lagged_ohe_data_test, 'all', rf, param_grid)
        short_rf = RandomForestRegressor(**gs_results.best_params_, random_state=SEED)
    short_rf, short_rf_train_rmse, short_rf_test_rmse = fit_eval(short_rf, short_lagged_ohe_data_train, short_lagged_ohe_data_test, 
                                                                 'all', 
                                                                 short_lagged_train_mean, short_lagged_train_std, fitted=short_rf_fitted)
    with open(short_rf_filepath,'wb') as f:
        pickle.dump(short_rf, f)
    
    long_rf_filepath = 'out/models/long_rf.pkl'
    long_rf_fitted = False
    if os.path.isfile(long_rf_filepath):
        with open(long_rf_filepath,'rb') as f:
            long_rf = pickle.load(f)
        long_rf_fitted = True
    else:
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        gs_results = run_grid_search(long_lagged_ohe_data_train, long_lagged_ohe_data_test, 'all', rf, param_grid)
        long_rf = RandomForestRegressor(**gs_results.best_params_, random_state=SEED)
    long_rf, long_rf_train_rmse, long_rf_test_rmse = fit_eval(long_rf, long_lagged_ohe_data_train, long_lagged_ohe_data_test, 
                                                              'all', 
                                                              long_lagged_train_mean, long_lagged_train_std, fitted=long_rf_fitted)
    with open(long_rf_filepath,'wb') as f:
        pickle.dump(long_rf, f)
    
    # Feature Set Generation

    top_k = 30

    all_features = long_lagged_ohe_data_train.columns.drop(['est'])
    importances = long_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in long_rf.estimators_], axis=0)
    top_features = pd.Series(importances, index=all_features).sort_values(ascending=False)[:top_k].sort_values(ascending=True)
    mdi_top_features = top_features.index[::-1]

    corr = short_lagged_ohe_data_train.corr()[['est']].sort_values(by='est', ascending=False)
    vmin = corr.min()
    vmax = corr.max()
    corr_thresh = corr.abs().sort_values('est', ascending=False).iloc[top_k+2]['est']
    corr = corr[corr['est'].abs() > corr_thresh]
    corr_features = corr[1:-1]

    X_train = short_lagged_ohe_data_train.drop(columns=['est'])
    y_train = short_lagged_ohe_data_train['est']
    X_test = short_lagged_ohe_data_test.drop(columns=['est'])
    y_test = short_lagged_ohe_data_test['est']
    ffs = SequentialFeatureSelector(LinearRegression(n_jobs=-1), k_features=top_k, forward=True, n_jobs=-1)
    ffs.fit(X_train, y_train)
    ffs_features = list(ffs.k_feature_names_)[::-1]

    # Lin-Reg

    param_grid = {'preproc__feature_selector__feature_names': [corr_features, ffs_features, mdi_top_features, all_features]}

    short_lr_filepath = 'out/models/short_lr.pkl'
    short_lr_fitted = False
    if os.path.isfile(short_lr_filepath):
        with open(short_lr_filepath,'rb') as f:
            short_lr = pickle.load(f)
        short_lr_fitted = True
    else:
        preproc = ColumnTransformer([('feature_selector', FeatureSelector(feature_names=[]), all_features)]
                                 ,remainder = 'drop')
        pl = Pipeline(steps=[('preproc', preproc), ('reg', LinearRegression(n_jobs=-1))])
        gs_results = run_grid_search(short_lagged_ohe_data_train, short_lagged_ohe_data_test, 'all', pl, param_grid)
        short_lr = gs_results.best_estimator_
    short_lr, short_lr_train_rmse, short_lr_test_rmse = fit_eval(short_lr, short_lagged_ohe_data_train, short_lagged_ohe_data_test, 
                                                                 'all', 
                                                                 short_lagged_train_mean, short_lagged_train_std, fitted=short_lr_fitted)
    with open(short_lr_filepath,'wb') as f:
        pickle.dump(short_lr, f)
    
    long_lr_filepath = 'out/models/long_lr.pkl'
    long_lr_fitted = False
    if os.path.isfile(long_lr_filepath):
        with open(long_lr_filepath,'rb') as f:
            long_lr = pickle.load(f)
        long_lr_fitted = True
    else:
        preproc = ColumnTransformer([('feature_selector', FeatureSelector(feature_names=[]), all_features)]
                                 ,remainder = 'drop')
        pl = Pipeline(steps=[('preproc', preproc), ('reg', LinearRegression(n_jobs=-1))])
        gs_results = run_grid_search(long_lagged_ohe_data_train, long_lagged_ohe_data_test, 'all', pl, param_grid)
        long_lr = gs_results.best_estimator_
    long_lr, long_lr_train_rmse, long_lr_test_rmse = fit_eval(long_lr, long_lagged_ohe_data_train, long_lagged_ohe_data_test, 
                                                            'all', 
                                                            long_lagged_train_mean, long_lagged_train_std, fitted=long_lr_fitted)
    with open(long_lr_filepath,'wb') as f:
        pickle.dump(long_lr, f)
        
    # LASSO
        
    short_lasso_filepath = 'out/models/short_lasso.pkl'
    short_lasso_fitted = False
    if os.path.isfile(short_lasso_filepath):
        with open(short_lasso_filepath,'rb') as f:
            short_lasso = pickle.load(f)
        short_lasso_fitted = True
    else:
        short_lasso = LassoCV(random_state=SEED)
    short_lasso, short_lasso_train_rmse, short_lasso_test_rmse = fit_eval(short_lasso, short_lagged_ohe_data_train, short_lagged_ohe_data_test, 
                                                                          'all', 
                                                                          short_lagged_train_mean, short_lagged_train_std, fitted=short_lasso_fitted)
    with open(short_lasso_filepath,'wb') as f:
        pickle.dump(short_lasso, f)
    
    long_lasso_filepath = 'out/models/long_lasso.pkl'
    long_lasso_fitted = False
    if os.path.isfile(long_lasso_filepath):
        with open(long_lasso_filepath,'rb') as f:
            long_lasso = pickle.load(f)
        long_lasso_fitted = True
    else:
        long_lasso = LassoCV(random_state=SEED)
    long_lasso, long_lasso_train_rmse, long_lasso_test_rmse = fit_eval(long_lasso, long_lagged_ohe_data_train, long_lagged_ohe_data_test, 
                                                                       'all', 
                                                                       long_lagged_train_mean, long_lagged_train_std, fitted=long_lasso_fitted)
    with open(long_lasso_filepath,'wb') as f:
        pickle.dump(long_lasso, f)
    
    # ARIMA

    short_arima_filepath = 'out/models/short_arima.pkl'
    if os.path.isfile(short_arima_filepath):
        with open(short_arima_filepath,'rb') as f:
            short_arima = pickle.load(f)
    else:
        short_arima = ARIMAForecast(short_data_train, 1, 1, 1)
        short_arima.train()
        with open(short_arima_filepath,'wb') as f:
            pickle.dump(short_arima, f)
    forecast = short_arima.forecast(short_data_test['year'].max())
    preds_labels = forecast.merge(short_data_test, on=['zip', 'year'], suffixes=('_pred', '_true'))
    short_arima_train_rmse = None
    short_arima_test_rmse = mean_squared_error(preds_labels['est_true'], preds_labels['est_pred'], squared=False)

    long_arima_filepath = 'out/models/long_arima.pkl'
    if os.path.isfile(long_arima_filepath):
        with open(long_arima_filepath,'rb') as f:
            long_arima = pickle.load(f)
    else:
        long_arima = ARIMAForecast(long_data_train, 1, 1, 1)
        long_arima.train()
        with open(long_arima_filepath,'wb') as f:
            pickle.dump(long_arima, f)
    forecast = long_arima.forecast(long_data_test['year'].max())
    preds_labels = forecast.merge(long_data_test, on=['zip', 'year'], suffixes=('_pred', '_true'))
    long_arima_train_rmse = None
    long_arima_test_rmse = mean_squared_error(preds_labels['est_true'], preds_labels['est_pred'], squared=False)

    # Windows

    IN_STEPS = 1
    OUT_STEPS = 1

    single_step_window = WindowGenerator(input_width=IN_STEPS,
                                        label_width=OUT_STEPS,
                                        shift=OUT_STEPS,
                                        train_df=long_ohe_data_train, 
                                        test_df=long_ohe_data_test,
                                        label_columns=['est'],
                                        batch_size=1)
    
    wide_window = WindowGenerator(input_width=5,
                                  label_width=5,
                                  shift=1,
                                  train_df=long_ohe_data_train, 
                                  test_df=long_ohe_data_test,
                                  label_columns=['est'],
                                  batch_size=1)

    short_data_train_by_zc_tf, short_data_test_by_zc_tf = split_by_zip_code(short_ohe_data_train, short_ohe_data_test, single_step_window, ignore_test=True)
    long_data_train_by_zc_tf, long_data_test_by_zc_tf = split_by_zip_code(long_ohe_data_train, long_ohe_data_test, single_step_window)
    
    wide_short_data_train_by_zc_tf, wide_short_data_test_by_zc_tf = split_by_zip_code(short_ohe_data_train, short_ohe_data_test, wide_window, ignore_test=True)
    wide_long_data_train_by_zc_tf, wide_long_data_test_by_zc_tf = split_by_zip_code(long_ohe_data_train, long_ohe_data_test, wide_window)

    # TF MODELS

    # Baseline

    class Baseline(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]
    column_indices = {name: i for i, name in enumerate(long_ohe_data_train.columns)}
    baseline = Baseline(label_index=column_indices['est'])
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.losses.MeanSquaredError()])
    
    # 5-in 5-out LSTM

    short_lstm_filepath = 'out/models/short_lstm_model.keras'
    try:
        short_lstm_model = tf.keras.models.load_model(short_lstm_filepath, compile=False)
        short_lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=[tf.keras.losses.MeanSquaredError()])
    except:
        short_lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(256, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
        losses, val_losses = compile_and_fit(short_lstm_model, wide_short_data_train_by_zc_tf, wide_short_data_test_by_zc_tf, MAX_EPOCHS, long_train_mean, long_train_std)
        short_lstm_model.save(short_lstm_filepath)

    long_lstm_filepath = 'out/models/long_lstm_model.keras'
    try:
        long_lstm_model = tf.keras.models.load_model(long_lstm_filepath, compile=False)
        long_lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=[tf.keras.losses.MeanSquaredError()])
    except:
        long_lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(256, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
        losses, val_losses = compile_and_fit(long_lstm_model, wide_long_data_train_by_zc_tf, wide_long_data_test_by_zc_tf, MAX_EPOCHS, long_train_mean, long_train_std)
        long_lstm_model.save(long_lstm_filepath)

    # Evaluate Models
        
    test_uni_window = WindowGenerator(input_width=1,
                                      label_width=1,
                                      shift=1,
                                      train_df=long_ohe_data_train, 
                                      test_df=long_ohe_data_test,
                                      label_columns=['est'],
                                      batch_size=1)
    test_uni_short_data_train_by_zc_tf, test_uni_short_data_test_by_zc_tf = split_by_zip_code(tf_short_ohe_data_train, tf_short_ohe_data_test, test_uni_window)
    test_uni_long_data_train_by_zc_tf, test_uni_long_data_test_by_zc_tf = split_by_zip_code(tf_long_ohe_data_train, tf_long_ohe_data_test, test_uni_window)

    models_to_test = [('baseline', baseline, baseline), 
                      ('lstm', short_lstm_model, long_lstm_model)]
    testing_scenarios = [('short-term', test_uni_short_data_train_by_zc_tf, test_uni_short_data_test_by_zc_tf),
                        ('long-term', test_uni_long_data_train_by_zc_tf, test_uni_long_data_test_by_zc_tf)]
    
    eval_info = []
    for model_name, short_model, long_model in models_to_test:
        model_eval = [model_name]
        for scenario_name, train_data, test_data in testing_scenarios:
            if scenario_name == 'short-term':
                model = short_model
            else:
                model = long_model
            train_rmse = evaluate_on_all_zip(model, train_data, long_train_mean, long_train_std)
            test_rmse = evaluate_on_all_zip(model, test_data, long_train_mean, long_train_std)
            model_eval += [train_rmse]
            model_eval += [test_rmse]
        eval_info += [model_eval]
    eval_info += [['lr', short_lr_train_rmse, short_lr_test_rmse,
                     long_lr_train_rmse, long_lr_test_rmse]]
    eval_info += [['rf', short_rf_train_rmse, short_rf_test_rmse,
                        long_rf_train_rmse, long_rf_test_rmse]]
    eval_info += [['lasso', short_lasso_train_rmse, short_lasso_test_rmse,
                            long_lasso_train_rmse, long_lasso_test_rmse]]
    eval_info += [['arima', short_arima_train_rmse, short_arima_test_rmse,
                            long_arima_train_rmse, long_arima_test_rmse]]
    
    evals_df = pd.DataFrame(eval_info, columns=['model', 
                                                'short-term train rmse', 'short-term test rmse',
                                                'long-term train rmse', 'long-term test rmse'])
    evals_df.to_csv('src/data/temp/model_evaluations.csv', index=False)
    
    # Vis Test Predictions

    long_train_trues = unstandardize_series(long_lagged_ohe_data_train, long_lagged_train_mean, long_lagged_train_std)[['year', 'est']].groupby('year').sum()
    long_test_trues = unstandardize_series(long_lagged_ohe_data_test, long_lagged_train_mean, long_lagged_train_std)[['year', 'est']].groupby('year').sum()
    long_test_trues = pd.concat([long_train_trues.iloc[[-1]], long_test_trues])

    long_rf_preds = long_lagged_ohe_data_test.copy()
    long_rf_preds['est'] = long_rf.predict(long_lagged_ohe_data_test.drop(columns=['est']))
    long_rf_preds = unstandardize_series(long_rf_preds, long_lagged_train_mean, long_lagged_train_std)[['year', 'est']]
    long_rf_preds = long_rf_preds.groupby('year').sum()
    long_rf_preds = pd.concat([long_train_trues.iloc[[-1]], long_rf_preds])

    long_lr_preds = long_lagged_ohe_data_test.copy()
    long_lr_preds['est'] = long_lr.predict(long_lagged_ohe_data_test.drop(columns=['est']))
    long_lr_preds = unstandardize_series(long_lr_preds, long_lagged_train_mean, long_lagged_train_std)[['year', 'est']]
    long_lr_preds = long_lr_preds.groupby('year').sum()
    long_lr_preds = pd.concat([long_train_trues.iloc[[-1]], long_lr_preds])

    long_lasso_preds = long_lagged_ohe_data_test.copy()
    long_lasso_preds['est'] = long_lasso.predict(long_lagged_ohe_data_test.drop(columns=['est']))
    long_lasso_preds = unstandardize_series(long_lasso_preds, long_lagged_train_mean, long_lagged_train_std)[['year', 'est']]
    long_lasso_preds = long_lasso_preds.groupby('year').sum()
    long_lasso_preds = pd.concat([long_train_trues.iloc[[-1]], long_lasso_preds])

    long_arima_preds = long_arima.forecast(2021).groupby('year')[['est']].sum()
    long_arima_preds = pd.concat([long_train_trues.iloc[[-1]], long_arima_preds])

    long_lstm_preds_raw = gen_sum_preds(long_lstm_model, test_uni_long_data_test_by_zc_tf, long_train_mean, long_train_std)
    long_lstm_preds = pd.DataFrame({'year': np.arange(2019, 2021 + 1),
                                    'est': long_lstm_preds_raw}).set_index('year')
    long_lstm_preds = pd.concat([long_train_trues.iloc[[-1]], long_lstm_preds])

    plt.figure(figsize=(5,5))
    plt.rcParams.update({'font.size': 10})
    plt.plot(long_train_trues[long_train_trues.index>2016], marker='o', c='tab:blue', label='trues')
    plt.plot(long_test_trues, marker='o', c='tab:blue', zorder=20)
    plt.plot(long_lr_preds, marker='o', c='tab:orange', label='lin-reg')
    plt.plot(long_lasso_preds, marker='o', c='tab:green', label='lasso')
    plt.plot(long_arima_preds, marker='o', c='tab:red', label='arima')
    plt.plot(long_rf_preds, marker='o', c='tab:purple', label='random-forest')
    plt.plot(long_lstm_preds, marker='o', c='tab:gray', label='lstm')
    plt.xticks(np.arange(2017, 2021+1))
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    final_forecasts_plot_filepath = 'out/plots/final_forecasts.jpg'
    plt.savefig(final_forecasts_plot_filepath, dpi=500, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    run()