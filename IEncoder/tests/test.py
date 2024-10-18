# # tests/test_encoders.py

# import numpy as np
# import pandas as pd
# import pytest
# from IEncoder.i_encoder import IEncoder

# # Fixture za sample data
# @pytest.fixture
# def sample_data():
#     data = pd.DataFrame({
#         'color': ['red', 'blue', 'green', 'blue', 'red'],
#         'size': ['S', 'M', 'L', 'M', 'S'],
#         'quantity': [10, 15, 7, 20, 10]
#     })
#     return data

# @pytest.fixture
# def encoder():
#     return IEncoder()

# def test_encoder_fit(sample_data, encoder):
#     encoder.fit(sample_data)
#     assert encoder.categories_ is not None
#     assert encoder.encoding_dict_ is not None
#     assert encoder.theta_arr_ is not None
#     assert encoder.n_features_in_ == sample_data.shape[1]
#     assert encoder.feature_names_in_ == ['x0', 'x1', 'x2']

# def test_transform(sample_data, encoder): ######
#     encoder.fit(sample_data)
#     transformed = encoder.transform(sample_data)
#     assert isinstance(transformed, np.ndarray)
#     assert transformed.shape == sample_data.shape
#     # Provjera da su nenumeričke kolone kodirane
#     assert not np.array_equal(transformed[:, 0], sample_data['color'].astype(float))
#     assert not np.array_equal(transformed[:, 1], sample_data['size'].astype(float))
#     # Numerička kolona ostaje ista
#     assert np.array_equal(transformed[:, 2], sample_data['quantity'])

# def test_inverse_transform(sample_data, encoder): ######
#     encoder.fit(sample_data)
#     transformed = encoder.transform(sample_data)
#     inversed = encoder.inverse_transform(transformed)
#     # Provjera da su originalne kategorije vraćene
#     pd.testing.assert_frame_equal(pd.DataFrame(inversed, columns=['x0', 'x1', 'x2']), sample_data)

# def test_fit_transform(sample_data, encoder):
#     transformed = encoder.fit_transform(sample_data)
#     assert isinstance(transformed, np.ndarray)
#     assert transformed.shape == sample_data.shape

# def test_get_feature_names_out(sample_data, encoder):
#     encoder.fit(sample_data)
#     feature_names = encoder.get_feature_names_out()
#     assert feature_names == ['x0', 'x1', 'x2']

# def test_set_params(sample_data, encoder):
#     # Iako vaša klasa nema parametre, ovaj test osigurava da set_params ne baca greške
#     encoder.set_params()
#     assert True  # Ako ne baci iznimku, test prolazi

# def test_get_params(sample_data, encoder): #####
#     params = encoder.get_params()
#     assert isinstance(params, dict)
#     assert 'categories_' in params
#     assert 'encoding_dict_' in params
#     assert 'theta_arr_' in params

# ctrl + a
# ctrl + '
