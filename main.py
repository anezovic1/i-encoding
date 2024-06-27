from sklearn.base import TransformerMixin
import numpy as np

class IEncoder(TransformerMixin):
    def __init__(self):
        # this dictionary contains the category as the key and the encoded value as the value
        self.encoding_dict_ = {}

    def fit(self, X, y=None):
        unique_categories = np.unique(X)
        n_categories = len(unique_categories)

        # the step is calculated in radians
        theta = 2 * np.pi / n_categories
        theta_arr = np.arange(0, 2 * np.pi, theta)
        theta_arr = np.round(theta_arr, 2)

        for idx, category in enumerate(unique_categories):
          self.encoding_dict_[category] = theta_arr[idx]

        return self

    def transform(self, X):
        return np.vectorize(self.encoding_dict_.get)(X)

    def fit_transform(self, X, y=None):
        # all categories in X are replaced with their encoded values
        return self.fit(X, y).transform(X)
    

X = np.array(['cat', 'dog', 'fish', 'cat', 'dog', 'fish'])

encoder = IEncoder()

encoded_X = encoder.fit_transform(X)

print("Originalni podaci:", X)
print("Kodirani podaci:", encoded_X)
