import numpy as np

class CustomTimeSeriesSplit:
    """
    Time Series cross-validator that provides train/test indices to split data into train/test sets.
    Time series data is split based on the 'year' column in the input X.
    """

    def __init__(self, n_splits=None):
        """
        Initialize CustomTimeSeriesSplit Object.

        Parameters
        ----------
        n_splits : int, optional
            Number of splits. If None, it will be inferred from the data.
        """
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : DataFrame
            Feature matrix. It should contain a 'year' column for time-based splitting.
        y : array-like, optional
            Target variable. Not used, present for compatibility.
        groups : array-like, optional
            Group labels for the samples. Not used, present for compatibility.

        Yields
        ------
        train : array-like
            The training set indices for that split.
        test : array-like
            The testing set indices for that split.
        """
        year_range = np.sort(X['year'].unique())
        min_year = year_range[0]
        
        self.n_splits = len(year_range) - 1
        
        for test_year in year_range[1:]:
            curr_range = np.arange(min_year, test_year)
            train = X[X['year'].apply(lambda year: year in curr_range)].index.to_numpy()
            test = X[X['year'] == test_year].index.to_numpy()
            
            yield train, test

    def get_n_splits(self, X, y, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Feature matrix. It must contain a 'year' column for time-based splitting.
        y : array-like
            Target variable. Not used, present for compatibility.
        groups : array-like, optional
            Group labels for the samples. Not used, present for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        year_range = np.sort(X['year'].unique())
        
        return len(year_range) - 1