import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class WindowGenerator():
    
    def __init__(self, input_width, label_width, shift,
                train_df, test_df,
                label_columns=None, batch_size=1):
        """
        A class to generate windows of time series data for training and evaluation.
        Calculates window parameters given input input design information.

        Parameters
        ----------
        input_width : int
            Width of the input window.
        label_width : int
            Width of the label window.
        shift : int
            Number of steps to shift the window for the next prediction.
        train_df : pandas.DataFrame
            DataFrame containing the training data.
        test_df : pandas.DataFrame
            DataFrame containing the test data.
        label_columns : list
            List of column names to be treated as labels
            Defaults to None, treating all features are labels
        batch_size : int, optional
            Batch size for the dataset, by default 1.
        """
        
        self.batch_size = batch_size
        
        # Store the raw data.
        self.train_df = train_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        """
        Returns a string representation of the WindowGenerator instance.
        """
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        """
        Splits the window into inputs and labels.

        Parameters
        ----------
        features : numpy.ndarray
            Input features.

        Returns
        -------
        tuple of tf.data.Datasets
            - inputs: tf.data.Datasets of input timesteps
            - labels: tf.data.Datasets of output timesteps
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels
    
    def make_dataset(self, data):
        """
        Creates a TensorFlow dataset from the given data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data.

        Returns
        -------
        tf.data.Dataset
            TensorFlow dataset.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,)
        ds = ds.map(self.split_window)
        return ds
    
    @property
    def train(self):
        """
        Property to get the training dataset as tf.data.Dataset.

        Returns
        -------
        tf.data.Dataset
            Training dataset.
        """
        return self.make_dataset(self.train_df)

    @property
    def example(self):
        """
        Get and cache an example batch of `inputs, labels` for plotting.

        Returns
        -------
        tuple
            Example batch of (inputs, labels).
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(self, model=None, plot_col='est', max_subplots=3):
        """
        Plots the inputs, labels, and predictions (if provided).

        Parameters
        ----------
        model : tf.keras.Model, optional
            Model for making predictions, by default None.
        plot_col : str, optional
            Column to plot, by default 'est'.
        max_subplots : int, optional
            Maximum number of subplots to display, by default 3.
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('year')