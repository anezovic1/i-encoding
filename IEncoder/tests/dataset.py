import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from IEncoder.iencoder import IEncoder

data = pd.read_csv('C:\\Users\\anida\\Downloads\\psi\\Airbnb\\train_users_2.csv\\train_users_2.csv')

print(data.head())

data.dropna(inplace=True)
label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])

i_encoder = IEncoder(handle_unknown='ignore', num_of_decimal_places=3, target_column='country_destination')

transformed = i_encoder.fit_transform(data)
print("data that is i encoded:\n", pd.DataFrame(transformed))

#print("\n\n\n", pd.DataFrame(transformed).iloc[7])

print(transformed)

print("data that is i encoded:\n", pd.DataFrame(transformed))



