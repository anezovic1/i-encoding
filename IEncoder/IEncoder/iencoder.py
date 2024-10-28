import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import copy
import warnings

class BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """
    def __init__(self, categories='auto', num_of_decimal_places=2, handle_unknown='error', target_column=None):
        self.categories_ = None
        self.encoding_dict_ = None
        self.theta_arr_ = None
        self.feature_names_in_ = None
        self.categories = categories
        self.num_of_decimal_places = num_of_decimal_places
        self.handle_unknown = handle_unknown
        self.target_column = target_column  
        self.label_encoder_ = LabelEncoder() if target_column else None

    def _check_finite(self, X):
        """
        Check if array contains NaN or infinite values.
        This check is only applicable for numeric data.
        """
        if hasattr(X, 'iloc'):  
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number): 
                    if np.any(np.isnan(X[col])) or np.any(np.isinf(X[col])):
                        raise ValueError(f"Column {col} contains NaN or infinity.")
        else:
            if np.issubdtype(X.dtype, np.number):
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    raise ValueError("Input contains NaN, infinity or a value too large for dtype.")

    def _check_X(self, X, force_all_finite=True):
        # this block executes when X is not a DataFrame
        if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
            X_temp = check_array(X, dtype=None)
            if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
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

            # this throws an exception if the DataFrame contains NaN values
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

        for i, column_name in enumerate(self.feature_names_in_):
            Xi = X_list[i]

            if column_name == self.target_column:
                if self.label_encoder_ is not None:
                    Xi = self.label_encoder_.fit_transform(Xi)  # target column encoded with LabelEncoder
                X_list[i] = Xi
                continue

            if not self._is_categorical(Xi):
                continue
            if self._skip_encoding(Xi):
                continue

            if self.categories != 'auto':
                unique_categories = self.categories[i]
            else:
                unique_categories = np.unique(Xi)

            n_categories = len(unique_categories)
            theta = 2 * np.pi / n_categories
            theta_arr = np.arange(0, 2 * np.pi, theta)
            theta_arr = np.round(theta_arr, self.num_of_decimal_places)

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

        for i, column_name in enumerate(self.feature_names_in_):
            Xi = X_list[i]

            if column_name == self.target_column:
                if self.label_encoder_ is not None:
                    Xi = self.label_encoder_.transform(Xi)  # target column encoded with LabelEncoder
                X_transformed[:, i] = Xi
                continue

            if not self._is_categorical(Xi):
                X_transformed[:, i] = Xi
                continue
            if self._skip_encoding(Xi):
                X_transformed[:, i] = Xi
                continue
            
            # encoding_dict_ already has key-value pairs
            encoding_dict = self.encoding_dict_[encoding_index] 
            categories = self.categories_[encoding_index]

            # check for unknown categories
            Xi = np.array(Xi)
            unknown_mask = ~np.isin(Xi, categories)

            if np.any(unknown_mask):
                if self.handle_unknown == "error":
                    raise ValueError(f"Unknown categories during transform")
                elif self.handle_unknown == "ignore":
                    # map unknown categories to np.nan
                    Xi_transformed = np.array([encoding_dict.get(xi, np.nan) for xi in Xi])
                elif self.handle_unknown == "warn":
                    warnings.warn(
                        f"Unknown categories during transform.",
                        UserWarning
                    )
                    # proceed as 'ignore'
                    Xi_transformed = np.array([encoding_dict.get(xi, np.nan) for xi in Xi])
                else:
                    raise ValueError(f"Invalid value for handle_unknown: {self.handle_unknown}")
            else:
                Xi_transformed = np.vectorize(encoding_dict.get)(Xi)

            X_transformed[:, i] = Xi_transformed
            encoding_index += 1

        return X_transformed

    def fit(self, X, y=None):
        """
        Fit the encoder to X.
        """
        valid_handle_unknown = {'error', 'ignore', 'warn'}
        if self.handle_unknown not in valid_handle_unknown:
            raise ValueError(
                f"Invalid value for 'handle_unknown': {self.handle_unknown}. "
                f"Allowed values are {valid_handle_unknown}."
            )
        return self._fit(X, y)

    def transform(self, X):
        """
        Transform X using the fitted encoder.
        """
        X_transformed = self._transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names_in_)

    def fit_transform(self, X, y=None):
        """
        Fit the encoder to X, then transform X.
        """
        X_transformed = self.fit(X, y).transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names_in_)

class IEncoder(BaseEncoder):
    """
    IEncoder that extends BaseEncoder1.
    """
    def __init__(self, categories='auto', num_of_decimal_places=2, handle_unknown='error', target_column=None):
        super().__init__(
            categories=categories, 
            num_of_decimal_places=num_of_decimal_places,
            handle_unknown=handle_unknown,
            target_column=target_column
        )

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

            # Handle unknown values (e.g., np.nan)
            unknown_mask = np.isnan(X_enc_i)
            if np.any(unknown_mask):
                X_enc_i = X_enc_i.copy()
                X_enc_i[unknown_mask] = np.nan

            indices = np.argmin(np.abs(X_enc_i[:, np.newaxis] - theta_arr), axis=1)
            X_inversed_i = categories[indices]

            # for unknowns, set to None
            if np.any(unknown_mask):
                X_inversed_i = X_inversed_i.astype(object)
                X_inversed_i[unknown_mask] = None

            X_inversed.append(X_inversed_i)

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
