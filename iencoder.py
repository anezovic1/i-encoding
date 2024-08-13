from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import copy

class IEncoder(TransformerMixin):
    def __init__(self):
        self.encoding_dict_ = {}

    def _is_categorical_or_integer(self, X):
        return X.dtype != 'float' and X.dtype != 'int64'

    def _skip_encoding(self, X):
        X_copy = copy.copy(X)

        try:
            _ = X_copy.astype(int)
            return True
        except ValueError:
            return False

    def fit(self, X, y=None):
        if not self._is_categorical_or_integer(X):
            return self
        if self._skip_encoding(X):
          return self

        unique_categories = np.unique(X)
        n_categories = len(unique_categories)
        theta = 2 * np.pi / n_categories
        theta_arr = np.arange(0, 2 * np.pi, theta)
        theta_arr = np.round(theta_arr, 2)

        for idx, category in enumerate(unique_categories):
            self.encoding_dict_[category] = theta_arr[idx]
        return self

    def transform(self, X):
        if not self._is_categorical_or_integer(X):
            return X
        if self._skip_encoding(X):
          return X
        return np.vectorize(self.encoding_dict_.get)(X)

    def fit_transform(self, X, y=None):
        if not self._is_categorical_or_integer(X):
            return X
        if self._skip_encoding(X):
          return X
        return self.fit(X, y).transform(X)