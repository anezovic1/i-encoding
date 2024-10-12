import pandas as pd
from IEncoder.i_encoder import IEncoder

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'quantity': [10, 15, 7, 20, 10]
})

encoder = IEncoder()

transformed = encoder.fit_transform(data)
print("data that is i encoded:\n", transformed)

inversed = encoder.inverse_transform(transformed)
print("inverse data:\n", inversed)
