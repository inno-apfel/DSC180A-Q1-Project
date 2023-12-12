import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')

class ARIMAForecast():
    """
    An object that represents a collection of ARIMA models that predicts employment counts, 
    where one model is trained for each ZIP Code in the provided data.
    """
    
    def __init__(self, data, n_lag_terms ,diff_order ,window_size):
        """
        Constructor for the ARIMAForecast class

        Parameters
        ----------
        data: pandas.DataFrame
            The data to train on
        n_lag_terms: int
            The number of lag terms to include in each ARIMA model
            Autoregressive component used for statsmodels.tsa.arima.model.ARIMA
        diff_order: int or List[int]
            The order of differencing to be done on data before training the ARIMA model
            Differencing component used for statsmodels.tsa.arima.model.ARIMA
        window_size: int or List[int]
            The width of each window that the ARIMA model looks at
            Moving average component used for statsmodels.tsa.arima.model.ARIMA
        """
        self.data = data
        self.models = {}
        self.n_lag_terms = n_lag_terms
        self.diff_order = diff_order
        self.window_size = window_size
        
    def train(self):
        """
        Trains X ARIMA models on data provided during object instantiation, 
        where X is the number of unique ZIP Codes in the data.
        """
        for zip_code in self.data['zip'].unique():
            # filter
            curr_data = self.data[self.data['zip']==zip_code][['year', 'emp']].set_index('year')
            start_time = curr_data.index[0]
            # train
            model = ARIMA(curr_data, order=(self.n_lag_terms ,self.diff_order ,self.window_size))
            try:
                results = model.fit()
                self.models[zip_code] = (results, start_time)
            except:
                pass
            
    def forecast(self, year):
        """
        Makes predictions for employment counts for every ZIP Code in the provided data, 
        for all years between the last year in the provided data and a given year (inclusive).

        Parameters
        ----------
        year: int
            The year to make predictions up to

        Returns
        -------
        pandas.DataFrame
            A table containing the forecast results. 
            Containing the columns: ['zip', 'year', 'emp']
        """
        preds = []
        # last year seen in the training set
        # used to calculate start range for forecast, to avoid predicting values in training set
        data_last_year = self.data['year'].max().year
        for zip_code, model_info in self.models.items():
            model, start_time = model_info
            # make predictions
            curr_pred = model.predict(data_last_year-start_time.year+1,year-start_time.year)
            # modify results into a df object
            curr_pred = curr_pred.to_frame().assign(zip=np.full(curr_pred.shape[0], zip_code)).reset_index()
            curr_pred = curr_pred.rename(columns={'index':'year', 0:'emp', 'predicted_mean':'emp'})
            # address issue where timestamp of some predictions is the number of years after the last year
            # in the training data rather than a timestamp object
            max_int = curr_pred[curr_pred['year'].apply(lambda x: type(x) == int)]['year'].max()
            curr_pred['year'] = curr_pred['year'].apply(lambda x: pd.Timestamp(str(year-max_int+x)) if (type(x) == int) else x)
            preds += [curr_pred]
            
        return pd.concat(preds, ignore_index=True).reset_index(drop=True)
            