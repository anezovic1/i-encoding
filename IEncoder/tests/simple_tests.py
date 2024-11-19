import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from IEncoder.iencoder import IEncoder

print("-------------------------------- 1. TEST --------------------------------\n\n")
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

print("Feature names:\n", encoder.get_feature_names_out())

print("-------------------------------- 2. TEST --------------------------------\n\n")
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

print("-------------------------------- 3. TEST --------------------------------\n\n")
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

print("-------------------------------- 4. TEST --------------------------------\n\n")

data_new_3 = pd.DataFrame({'Category': ['A', 'B', 'C', 'A', 'B']})

one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data_new_3[['Category']])
encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Category']))
print("one-hot encoded data:\n", encoded_df)

i_encoder = IEncoder()
i_encoded = encoder.fit_transform(data_new_3[['Category']])
i_encoded_df = pd.DataFrame(i_encoded, columns=i_encoder.get_feature_names_out(['Category']))
print("i encoded data:\n", i_encoded_df)

print("-------------------------------- 5. TEST --------------------------------\n\n")  

data_new_4 = pd.DataFrame({'Category': [1, 2, np.nan, 2, 1]}) 

one_hot_encoder = OneHotEncoder(sparse_output=False)
try:
    one_hot_encoded = one_hot_encoder.fit_transform(data_new_4[['Category']])
    print("one-hot encoded data with NaN:\n", one_hot_encoded)
except Exception as e:
    print(e)

i_encoder = IEncoder()
try:
    i_encoded = i_encoder.fit_transform(data_new_4[['Category']])
    print("i encoded data with NaN:\n", i_encoded)
except Exception as e:
    print(e)

print("-------------------------------- 6. TEST --------------------------------\n\n")  

data_new_4 = pd.DataFrame({'A': [1, 2, np.nan, 1, 2], 
                           'B': [5, 6, 5, 3, 6]
                           }) 

i_encoder = IEncoder()
try:
    print("i encoded data with NaN:\n")
    i_encoded = i_encoder.fit_transform(data_new_4[['A', 'B']])
except Exception as e:
    print(e)

print("-------------------------------- 7. TEST --------------------------------\n\n")  

data_new_4 = pd.DataFrame({'A': [1, 2, 5, 1, 2], 
                           'B': [5, 6, np.inf, 3, 6]
                           }) 

i_encoder = IEncoder()
try:
    print("i encoded data with NaN:\n")
    i_encoded = i_encoder.fit_transform(data_new_4[['A', 'B']])
except Exception as e:
    print(e)

print("-------------------------------- 8. TEST --------------------------------\n\n")  

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

categories = [
    ['blue', 'green', 'red'],  
    ['S', 'M', 'L']         
]

encoder = IEncoder(categories=categories)
transformed = encoder.fit_transform(data)

print("Transformed data:\n", transformed)
print(encoder.get_params())

print("-------------------------------- 9. TEST --------------------------------\n\n")  

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

categories = [
    ['blue', 'green', 'red'],  
    ['S', 'M', 'L']         
]

encoder = IEncoder(categories=categories, num_of_decimal_places=5)
transformed = encoder.fit_transform(data)

print("Transformed data:\n", transformed)

print("-------------------------------- 10. TEST --------------------------------\n\n")  

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

encoder = IEncoder(handle_unknown='warn')
encoder.fit(data)
transformed = encoder.transform(data)
print("Transformed data:\n", transformed)

new_data = pd.DataFrame({
    'color': ['red', 'yellow'],  
    'size': ['S', 'XL']          
})
transformed_new = encoder.transform(new_data)
print("Transformed new data:\n", transformed_new)

inversed_new = encoder.inverse_transform(transformed_new)
print("Inverse transformed new data:\n", inversed_new)

print("-------------------------------- 11. TEST --------------------------------\n\n")  

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

encoder = IEncoder(categories='auto', num_of_decimal_places=2, handle_unknown='error')

print("Initial parameters:")
print(encoder.get_params())

encoder.set_params(num_of_decimal_places=4, handle_unknown='ignore')

print("\nUpdated parameters:")
print(encoder.get_params())

encoder.fit(data)
transformed_data = encoder.transform(data)

print("\nTransformed data with updated parameters:")
print(transformed_data)

