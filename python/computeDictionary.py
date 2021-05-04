import pickle
from python.getDictionary import get_dictionary, CORNER_HARRIS, RANDOM

# meta = pickle.load(open('../data/traintest.pkl', 'rb'))
# train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

K = 100
alpha = 50
method = CORNER_HARRIS

# Harris Points
dictionary = get_dictionary(train_imagenames, alpha, K, method)
pickle.dump(dictionary, open("dictionaryHarris.pkl", "wb"))

# Random Points
method = RANDOM
dictionary = get_dictionary(train_imagenames, alpha, K, method)
pickle.dump(dictionary, open("dictionaryRandom.pkl", "wb"))

# ----------------------------------------------