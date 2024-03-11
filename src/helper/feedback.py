import pandas as pd
import numpy as np

import tensorflow as tf

class FeedBack(tf.keras.Model):

    def __init__(self, units, out_steps, num_features):
        """
        Initializes the Autoregressive Feedback LSTM model.

        Parameters
        ----------
        units : int
            The number of LSTM units
        out_steps : int
            Number of future timestamps to predict at inference
        num_features : int
            Number of features in the input data.
        """
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        """
        Initializes the LSTMs hidden state using given inputs

        Parameters
        ----------
        inputs : tf.Tensor
            Input data in the shape (batch, time, features).

        Returns
        -------
        prediction: tf.Tensor
            Predicted output for the given input
        state: tf.Tensor
            Hidden state of the LSTM after processing input
        """
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    
    def call(self, inputs, training=None):
        """
        Generates predictions for the given input data

        Parameters
        ----------
        inputs : tf.Tensor
            Input data in the shape (batch, time, features)
        training : bool
            Whether or not given inputs are used to fit the model

        Returns
        -------
        predictions: tf.Tensor
            Predictions for the input data in the shape (batch, out_steps, num_features).
        """
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
