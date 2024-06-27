import numpy as np
from iencoder import IEncoder

# TEST 1
X = np.array(['cat', 'dog', 'fish', 'cat', 'dog', 'fish'])
encoder = IEncoder()
encoded_X = encoder.fit_transform(X)
print("Original data:", X)
print("Encoded data:", encoded_X)

# TEST 2
X = np.array(['1', '2', '3', '1', '2', '3'])
encoder = IEncoder()
encoded_X = encoder.fit_transform(X)
print("Original data:", X)
print("Encoded data:", encoded_X)

# TEST 3
X = np.array([1, 2, 5, 10, 11])
encoder = IEncoder()
encoded_X = encoder.fit_transform(X)
print("Original data:", X)
print("Encoded data:", encoded_X)