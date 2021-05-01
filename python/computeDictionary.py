import pickle
from python.getDictionary import get_dictionary


meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

K = 100
method = None

dictionary = get_dictionary(train_imagenames, 50, K, method)

pickle.dump(dictionary, open("dictionaryHarris.pkl", "wb"))
# pickle.dump(dictionary, open("dictionaryRandom.pkl", "wb"))

# ----------------------------------------------