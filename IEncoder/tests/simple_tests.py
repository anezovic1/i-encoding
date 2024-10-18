import pandas as pd
import numpy as np
from IEncoder.i_encoder import IEncoder

print("-------------------------------- TEST --------------------------------\n\n")
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

print("-------------------------------- TEST --------------------------------\n\n")
########################### second example ########################### 
data_new = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        'NewColumn': ['1', '2', '1', '3', '2'], # only nominal
        }

encoder = IEncoder()

df = pd.DataFrame(data_new)
print("employee data:\n", df)

transformed = encoder.fit_transform(df)
print("data that is i encoded:\n", transformed)

print("-------------------------------- TEST --------------------------------\n\n")
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

print("-------------------------------- TEST --------------------------------\n\n")
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

print("-------------------------------- TEST --------------------------------\n\n")  # ????
########################### fifth example ########################### 

data_new_4 = pd.DataFrame({'Category': ['A', 'B', np.nan, 'A', 'B']})

one_hot_encoder = OneHotEncoder(sparse_output=False)
try:
    one_hot_encoded = one_hot_encoder.fit_transform(data_new_4[['Category']])
    print("one-hot encoded data with NaN:\n", one_hot_encoded)
except Exception as e:
    print("izuzetak bacen")
    print(e)

i_encoder = IEncoder()
try:
    i_encoded = i_encoder.fit_transform(data_new_4[['Category']])
    print("i encoded data with NaN:\n", i_encoded)
except Exception as e:
    print("izuzetak bacen")
    print(e)


# print("-------------------------------- TEST --------------------------------\n\n")  # ????
# ########################### sixth example ########################### 

# enc = OneHotEncoder(handle_unknown='ignore')
# X = [['Male', 1], ['Female', 3], ['Female', 2]]
# enc.fit(X)
# OneHotEncoder(handle_unknown='ignore')
# print(enc.categories_)
# print(enc.transform([['Female', 1], ['Male', 4]]).toarray())
# print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
# print(enc.get_feature_names_out(['gender', 'group']))

# encoder = IEncoder()
# X = [['Male', 1], ['Female', 3], ['Female', 2]]
# encoder.fit(X)

# print(encoder.categories_)
# print(encoder.transform([['Female', 1], ['Male', 4]]).toarray())
# print(encoder.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
# print(encoder.get_feature_names_out(['gender', 'group']))
