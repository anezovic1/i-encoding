import pandas as pd
from IEncoder.i_encoder import IEncoder

# first example
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

# second example
data_new = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        'NewColumn': ['1', '2', '1', '3', '2'],
        }

encoder = IEncoder()

df = pd.DataFrame(data_new)
print("employee data:\n", df)

transformed = encoder.fit_transform(df)
print("data that is i encoded:\n", transformed)