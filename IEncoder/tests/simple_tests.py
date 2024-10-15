import pandas as pd
from IEncoder.i_encoder import IEncoder

########################### first example ########################### 
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

########################### second example ########################### 
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

########################### third example ########################### 
data_new_2 = pd.DataFrame({'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        'NewColumn': [True, True, False, False, False],
    })

encoder = IEncoder()

transformed = encoder.fit_transform(data_new_2)
print("data that is i encoded:\n", transformed)

inversed = encoder.inverse_transform(transformed)
print("inverse data:\n", inversed)

########################### fourth example ########################### 
from sklearn.preprocessing import OneHotEncoder

data_new_3 = pd.DataFrame({'Category': ['A', 'B', 'C', 'A', 'B']})

one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data_new_3[['Category']])
encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Category']))
print("one-hot encoded data:\n", encoded_df)

i_encoder = IEncoder()
i_encoded = encoder.fit_transform(data_new_3[['Category']])
i_encoded_df = pd.DataFrame(i_encoded, columns=i_encoder.get_feature_names_out(['Category']))
print("i encoded data:\n", i_encoded_df)
