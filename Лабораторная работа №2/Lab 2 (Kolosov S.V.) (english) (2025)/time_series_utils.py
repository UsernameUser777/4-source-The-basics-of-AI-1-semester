"""
time_series_utils.py

Helper module for processing time series data.
Contains the TimeSeriesDataset class for loading, normalizing,
creating windows, and splitting data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset:
    """
    Class for processing time series: normalization, creating windows, splitting data.
    """

    def __init__(self, filepath=None, data=None, depth=5, horizon=1, feature_range=(-1, 1)):
        """
        Initialize the data processor.

        Args:
            filepath (str, optional): Path to the dataset CSV file.
                                      Should contain a 'Value' or 'Output' column.
            data (array-like, optional): Directly provide the time series (if filepath is not specified).
            depth (int): Depth of embedding (n) for the window method.
            horizon (int): Forecasting horizon (m).
            feature_range (tuple): Range for normalization (min, max).
        """
        if filepath:
            df = pd.read_csv(filepath)
            # Assume the column with original data is called 'Value' or 'Output'
            # Can be adapted to specific file format
            if 'Value' in df.columns:
                self.raw_data = df['Value'].values.astype(np.float32)
            elif 'Output' in df.columns:
                # For files with pre-made windows, extract 'Output' as the original series
                # This is less reliable than using 'Value' from all_datasets_combined
                self.raw_data = df['Output'].values.astype(np.float32)
                # Restore the beginning of the series from the first window's Input
                first_window_inputs = df.iloc[0, :depth].values.astype(np.float32)
                self.raw_data = np.concatenate([first_window_inputs, self.raw_data])
            else:
                raise ValueError(f"The file {filepath} does not contain a 'Value' or 'Output' column.")
        elif data is not None:
            self.raw_data = np.array(data, dtype=np.float32)
        else:
            raise ValueError("Either filepath or data must be provided.")

        self.depth = depth
        self.horizon = horizon
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.normalized_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_process(self):
        """
        Perform full preprocessing:
        normalization, creating windows, splitting into training and test sets.
        """
        # Normalize
        self.normalized_data = self.scaler.fit_transform(self.raw_data.reshape(-1, 1)).flatten()

        # Create windows
        self.X, self.y = self.create_windows(self.normalized_data, self.depth, self.horizon)

        # Split into training and test sets (70/30)
        split_idx = int(len(self.X) * 0.7)
        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]

    def create_windows(self, data, depth, horizon):
        """
        Creates windows for training from a 1D time series.

        Args:
            data (np.array): Normalized 1D array.
            depth (int): Depth of embedding (n).
            horizon (int): Forecasting horizon (m).

        Returns:
            tuple: (X, y) - feature arrays and target values.
        """
        X, y = [], []
        for i in range(len(data) - depth - horizon + 1):
            X.append(data[i:i + depth])
            y.append(data[i + depth:i + depth + horizon][0])  # Only the first value after the window
        return np.array(X), np.array(y)

    def split_data(self, train_ratio=0.7):
        """
        Split data into training and test sets.
        (Alternative to calling load_and_process if data is already created)
        Args:
            train_ratio (float): Training set ratio.
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("Data (X, y) not created. Call create_windows or load_and_process first.")
        split_idx = int(len(self.X) * train_ratio)
        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_features_and_targets(self):
        """
        Return the created windows (X, y).
        """
        if self.X is None or self.y is None:
            raise ValueError("Data (X, y) not created. Call create_windows or load_and_process.")
        return self.X, self.y

    def inverse_normalize(self, value):
        """
        Revert value from normalized scale back to original.
        """
        return self.scaler.inverse_transform(np.array(value).reshape(-1, 1))

    def get_raw_data(self):
        """
        Return the original data.
        """
        return self.raw_data

    def get_normalized_data(self):
        """
        Return the normalized data.
        """
        return self.normalized_data
