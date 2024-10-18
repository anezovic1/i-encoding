import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import copy

class BaseEncoder1(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """

    def __init__(self):
        self.categories_ = None
        self.encoding_dict_ = None
        self.theta_arr_ = None
        self.feature_names_in_ = None

    def _check_finite(self, X):
        """
        Check if array contains NaN or infinite values.
        This check is only applicable for numeric data.
        """
        if np.issubdtype(X.dtype, np.number):
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Input contains NaN, infinity or a value too large for dtype.")

    def _check_X(self, X, force_all_finite=True):
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            X_temp = check_array(X, dtype=None)
            
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object)
            else:
                X = X_temp

            if force_all_finite:
                self._check_finite(X)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.to_list()
        else:
            self.feature_names_in_ = ['Column%d' % i for i in range(X.shape[1])]

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            Xi = check_array(Xi, ensure_2d=False, dtype=None)

            if force_all_finite:
                self._check_finite(Xi)
            
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _is_categorical(self, X):
        return X.dtype.kind in {'O', 'U', 'S'} or np.issubdtype(X.dtype, np.integer)

    def _skip_encoding(self, X):
        X_copy = copy.copy(X)
        try:
            _ = X_copy.astype(int)
            return True
        except ValueError:
            return False

    def _fit(self, X, y=None):
       
        X_list, n_samples, n_features = self._check_X(X)
        self.n_features_in_ = n_features
        self.categories_ = []
        self.encoding_dict_ = []
        self.theta_arr_ = []

        for i in range(n_features):
            Xi = X_list[i]
            if not self._is_categorical(Xi):
                continue
            if self._skip_encoding(Xi):
                continue

            unique_categories = np.unique(Xi)
            n_categories = len(unique_categories)
            theta = 2 * np.pi / n_categories
            theta_arr = np.arange(0, 2 * np.pi, theta)
            theta_arr = np.round(theta_arr, 2)

            encoding_dict = {category: theta_val for category, theta_val in zip(unique_categories, theta_arr)}
            self.categories_.append(unique_categories)
            self.encoding_dict_.append(encoding_dict)
            self.theta_arr_.append(theta_arr)

        return self

    def _transform(self, X):

        check_is_fitted(self, ['categories_', 'encoding_dict_'])
        X_list, n_samples, n_features = self._check_X(X)
        X_transformed = np.zeros((n_samples, n_features))

        encoding_index = 0

        for i in range(n_features):
            Xi = X_list[i] 
            if not self._is_categorical(Xi):
                X_transformed[:, i] = Xi
                continue
            if self._skip_encoding(Xi):
                X_transformed[:, i] = Xi
                continue
            encoding_dict = self.encoding_dict_[encoding_index] # encoding_dict_ already has key-value pairs
            X_transformed[:, i] = np.vectorize(encoding_dict.get)(Xi)

            encoding_index = encoding_index + 1

        return X_transformed

    def fit(self, X, y=None):
        """
        Public fit method that calls the internal _fit method.
        """
        return self._fit(X, y)

    def transform(self, X):
        """
        Public transform method that calls the internal _transform method.
        """
        return self._transform(X)

    def fit_transform(self, X, y=None):
        """
        Combines fit and transform methods for efficiency.
        """
        return self.fit(X, y).transform(X)


class IEncoder(BaseEncoder1):
    """
    I Encoder that extends BaseEncoder1.
    """

    def __init__(self):
        super().__init__()

    def inverse_transform(self, X_encoded):
        """
        Inverse transform method to convert encoded data back to original categories.
        """
        check_is_fitted(self, ['theta_arr_', 'categories_'])
        X_encoded = np.asarray(X_encoded)
        if X_encoded.ndim == 1:
            X_encoded = X_encoded.reshape(-1, 1)
        n_samples, n_features = X_encoded.shape
        X_inversed = []

        for i in range(n_features):
            if i >= len(self.theta_arr_):
                X_inversed.append(X_encoded[:, i])
                continue
            theta_arr = self.theta_arr_[i]
            categories = self.categories_[i]
            X_enc_i = X_encoded[:, i]
            indices = np.argmin(np.abs(X_enc_i[:, np.newaxis] - theta_arr), axis=1)
            X_inversed.append(categories[indices])

        return np.array(X_inversed).T

    def get_feature_names_out(self, input_features=None):
        """
        Returns output feature names for transformation.
        """
        check_is_fitted(self, 'feature_names_in_')
        if input_features is None:
            input_features = self.feature_names_in_
        return input_features

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        return super().set_params(**params)