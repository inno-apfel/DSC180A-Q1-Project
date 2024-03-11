from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_names):
        """
        Transformer for selecting specific features from a dataset.

        Parameters
        ----------
        feature_names : list
            List of strings representing the names of the features to be selected.

        """
        self.feature_names = feature_names

    def fit(self, X, y=None):
        """
        Fit method required by TransformerMixin interface.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : array-like, default=None
            Ignored variable. Included for compatibility.

        Returns
        -------
        self : FeatureSelector
            Returns the estimator itself.
        """
        return self
    
    def transform(self, X):
        """
        Transforms the input data by selecting the specified features.

        Parameters
        ----------
        X : DataFrame
            Input features.

        Returns
        -------
        DataFrame
            DataFrame containing only the selected features.
        """
        selected_features = [col for col in X.columns if any(name in col for name in self.feature_names)]
        return X[selected_features]