import pickle
from getDictionary import get_dictionary
import numpy as np


# meta = pickle.load (open('../data/traintest.pkl', 'rb'))

# test_imagenames = meta['test_imagenames']


# -----fill in your implementation here --------


# ----------------------------------------------



visionHarris = pickle.load(open("visionHarris.pkl", "rb"))
visionRandom = pickle.load(open("visionRandom.pkl", "rb"))

print(visionHarris['dictionary'].shape)
# print(visionHarris[2].shape)
# print(visionRandom[2].shape)
