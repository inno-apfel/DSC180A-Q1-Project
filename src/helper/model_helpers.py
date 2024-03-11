import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

sys.path.insert(0, 'src/helper')
from custom_tcsv import CustomTimeSeriesSplit


def train_test_split_by_year(data, end_year):
    """
    Splits a dataset into training and testing sets based on a specified end year.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be split, should contain a column named 'year' indicating the year of each data point.

    end_year : int
        The year at which to split the data into training and testing sets.
        Data points with years less than or equal to 'end_year' will be in the training set,
        while those with years greater than 'end_year' will be in the testing set.

    Returns
    -------
    data_train : pandas.DataFrame
        The subset of data with years less than or equal to 'end_year', forming the training set.

    data_test : pandas.DataFrame
        The subset of data with years greater than 'end_year', forming the testing set.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'year': [2010, 2011, 2012, 2013, 2014, 2015],
                             'value': [1, 2, 3, 4, 5, 6]})
    >>> train, test = train_test_split_by_year(data, 2012)
    >>> train
       year  value
    0  2010      1
    1  2011      2
    2  2012      3
    >>> test
       year  value
    3  2013      4
    4  2014      5
    5  2015      6
    """
    data_train = data[data['year'] <= end_year]
    data_test = data[data['year'] > end_year]
    return data_train, data_test

def standardize_data(data_train, data_test):
    """
    Standardizes the training and testing data using mean and standard deviation calculated from the training data.

    Parameters
    ----------
    data_train : pandas.DataFrame
        DataFrame containing training data.
    data_test : pandas.DataFrame
        DataFrame containing testing data.

    Returns
    -------
    data_train_standardized : pandas.DataFrame
        Standardized training data.
    data_test_standardized : pandas.DataFrame
        Standardized testing data.
    statistics : tuple
        Tuple containing mean and standard deviation used for standardization.
        First element is a pandas.Series representing the mean,
        and the second element is a pandas.Series representing the standard deviation.
        The 'zip' field in both Series is set to 0 for mean and 1 for standard deviation.
    """
    train_mean = data_train.mean()
    train_mean['zip'] = 0
    train_std = data_train.std()
    train_std['zip'] = 1
    data_train_standardized = (data_train - train_mean) / train_std
    data_test_standardized = (data_test - train_mean) / train_std
    return data_train_standardized, data_test_standardized, (train_mean, train_std)

def unstandardize_series(ser, mean, std):
    """
    Unstandardizes a series of data points using the given mean and standard deviation.

    Parameters
    ----------
    ser : pandas.Series
        Series of standardized data points.
    mean : float
        Mean value used for standardization.
    std : float
        Standard deviation value used for standardization.

    Returns
    -------
    pandas.Series
        Series of unstandardized data points.
    """
    return (ser*std)+mean

def convert_to_ohe(data_train, data_test):
    """
    Converts 'zip' feature into one-hot encoded representation.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Training dataset
    data_test : pandas.DataFrame
        Test dataset

    Returns
    -------
    data_ohe_train : pandas.DataFrame
        One-hot encoded training dataset.
    data_ohe_test : pandas.DataFrame
        One-hot encoded test dataset.

    Notes
    -----
    This function expects the categorical features to be present in the 'zip' column of the input dataframes.
    """
    
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
    """
    Splits the input data into training and testing datasets based on zip codes.

    Parameters
    ----------
    data_train : DataFrame
        Training dataset.
    data_test : DataFrame
        Testing dataset.
    window : window_generator.WindowGenerator
        A WindowGenerator object to convert datasets to tf.datasets using
    ignore_test : bool, optional
        If True, the testing dataset will be ignored, and only the training dataset will be processed. 

    Returns
    -------
    tuple of dict
        A tuple containing two dictionaries:
        - data_train_by_zc_tf: Dictionary mapping zip codes to training datasets.
        - data_test_by_zc_tf: Dictionary mapping zip codes to testing datasets
    """
    
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

def fit_eval(model, data_train, data_test, included_feats, train_mean, train_std, fitted=False):
    """
    Fits the given model on training data and evaluates its performance on both training and testing data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        An sklearn model to be trained and evaluated.

    data_train : pandas.DataFrame
        Training dataset containing features and target variable.

    data_test : pandas.DataFrame
        Testing dataset containing features and target variable.

    included_feats : list
        List of feature names to be included in training and evaluation.
        If 'all', all columns except 'est' are included as features.

    train_mean : pandas.Series
        Mean values used for standardization during training.

    train_std : pandas.Series
        Standard deviation values used for standardization during training.

    fitted : bool, optional
        If True, assumes the model is already fitted on data_train and skips fitting step.
        Default is False.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - trained_model : sklearn.base.BaseEstimator
            The trained model.
        - train_rmse : float
            Root Mean Squared Error (RMSE) of the model predictions on the training data.
        - test_rmse : float
            Root Mean Squared Error (RMSE) of the model predictions on the testing data.

    Notes
    -----
    The function assumes that both data_train and data_test have columns 'est' representing the target variable.
    This function requires the unstandardize_series function and mean_squared_error function from sklearn.metrics.
    """
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
    """
    Runs grid search with cross-validation to find the best hyperparameters for the given model.

    Parameters
    ----------
    data_train : pandas DataFrame
        Training dataset containing both features and target variable.
    data_test : pandas DataFrame
        Testing dataset containing both features and target variable.
    included_feats : str or list of str
        Names of the columns/features to be included in the model. If 'all', all columns except 'est' will be used.
    model : scikit-learn estimator object
        Model for which hyperparameters will be optimized.
    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object with the best hyperparameters found.
    """
    if included_feats == 'all':
        included_feats = data_train.columns.drop(['est'])
    X_train = data_train[included_feats]
    y_train = data_train['est']
    grid_search = GridSearchCV(model, param_grid,cv = CustomTimeSeriesSplit(), scoring = 'neg_root_mean_squared_error', n_jobs = -1)
    grid_search.fit(X_train, y_train)
    return grid_search

def compile_and_fit(model, data_train_by_zc, data_test_by_zc, num_epochs, train_mean, train_std):
    """
    Compiles the provided model and fits it to the training data for a num_epochs epochs.

    Parameters
    ----------
    model : tf.Model
        The tensorflow model to compile and fit.
    data_train_by_zc : dict
        A dictionary where keys are zip codes and values are training data.
    data_test_by_zc : dict
        A dictionary where keys are zip codes and values are test data.
    num_epochs : int
        Number of epochs for training.
    train_mean : pandas.Series
        Mean values used for standardization during training.
    train_std : pandas.Series
        Standard deviation values used for standardization during training.

    Returns
    -------
    losses : list
        List of root mean squared losses for each epoch on the training data.
    val_losses : list
        List of root mean squared losses for each epoch on the validation data.

    Notes
    -----
    This function compiles the model with MeanSquaredError loss, Adam optimizer,
    and MeanSquaredError as the metric. It then trains the model for the specified
    number of epochs using the provided training data by zip code, evaluating on
    the same data for simplicity.

    During training, if the difference in losses between consecutive epochs is
    less than 0.1 for two consecutive epochs, the patience for early stopping
    decreases by one. Training stops early if patience reaches 0.

    The losses and validation losses are collected for each epoch, unstandardized,
    and stored in respective lists for further analysis.
    """
    
    KERAS_VERBOSITY = 0
    patience = 4
    losses = []
    val_losses = []
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.losses.MeanSquaredError()])
    for epoch in np.arange(num_epochs):
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

def evaluate_on_all_zip(model, data_by_zc, train_mean, train_std):
    """
    Evaluates a tensorflow model on data generateed from the function split_by_zip_code and returns the root mean square error (RMSE).

    Parameters
    ----------
    model : tf.Model
        Trained model to be evaluated.
    data_train_by_zc : dict
        Dictionary containing training data grouped by ZIP code.
    train_mean : pd.Series
        Series containing mean values used for standardization during training.
    train_std : pd.Series
        Series containing standard deviation values used for standardization during training.

    Returns
    -------
    float
        Root mean square error (RMSE) of the model evaluated on all ZIP codes in the training data.
    """
    total = 0
    i = 0
    for zc in data_by_zc.keys():
        total += unstandardize_series(model.evaluate(data_by_zc[zc], verbose=0)[0], 
                                      train_mean['est'], train_std['est'])
        i += 1
    return np.sqrt(total/i)

def sum_auto_wide_plot_model(model, wide_data, window, extra_steps, train_mean, train_std):
    """
    Aggregates (sum) the predictions of the given tensorflow model across all ZIP Codes.

    Parameters
    ----------
    model : tf.Model
        The trained model to generate predictions.
    wide_data : dict
        A dictionary containing data for multiple ZIP Codes.
        Each key represents a ZIP code, and its value is a tf.dataset.
    window : window_generator.WindowGenerator
        A WindowGenerator object providing column indices.
    extra_steps : int
        How many timesteps to generate.
    train_mean : pd.Series
        A Series containing mean values for standardization used during training.
    train_std : pd.Series
        A Series containing standard deviation values for standardization used during training.

    Returns
    -------
    total_inputs : numpy.ndarray
        The total inputs (unstandardized) across all zones and batches.
    total_preds : numpy.ndarray
        The total predictions (unstandardized) across all zones and batches.
    """
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

def gen_sum_preds(model, data_test, train_mean, train_std):
    """
    Generate and aggregate (sum) predictions for a tensorflow model across all ZIP codes.

    Parameters
    ----------
    model : tf.Model
        Trained tensorflow model for prediction.

    data_test : dict
        Dictionary containing test data for different zip codes.

    train_mean : pd.Series
        Series containing mean values from training set.

    train_std : pd.Series
        Series containing standard deviation values from training set.

    Returns
    -------
    numpy.ndarray
        Summed predictions for the region.

    """
    total_preds = None
    for zc in data_test.keys():
        curr_preds = model.predict(data_test[zc], verbose=0)[:, 0, 0]
        curr_preds = unstandardize_series(curr_preds, train_mean['est'], train_std['est'])
        if total_preds is None:
            total_preds = curr_preds
        else:
            total_preds += curr_preds
    return total_preds