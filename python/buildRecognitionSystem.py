import pickle
import cv2
import numpy as np
from createFilterBank import create_filterbank

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
train_labels = meta['train_labels']

dictionaryHarris = pickle.load(open("dictionaryHarris.pkl", "rb"))
dictionaryRandom = pickle.load(open("dictionaryRandom.pkl", "rb"))


# TODO: In each pickle store a dictionary that contains:
# 1. dictionary:  your visual word dictionary, a matrix of size K x 3n
# 2. filterBank:  filter bank used to produce the dictionary.  This is an array of image filters
# 3. trainFeatures:  T x K matrix containing all of the histograms of visual words of the T training images in the data set.
# 4. trainLabels:  T x 1 vector containing the labels of each training image.

visionRandom = {0: None, 1: None, 2: None, 3: None}
visionHarris = {0: None, 1: None, 2: None, 3: None}

# 1. dictionary:
visionRandom[0] = dictionaryRandom
visionHarris[0] = dictionaryHarris

# 2. filterBank:
fb = create_filterbank()
visionRandom[1] = fb
visionHarris[1] = fb

# 3. trainFeatues:
# visionRandom[2] = train_imagenames
# visionHarris[2] = train_imagenames

# 4.  trainLabels:
visionRandom[3] = train_labels
visionHarris[3] = train_labels

# Save to pickle file
pickle.dump(visionRandom, open("visionRandom.pkl", "wb"))
pickle.dump(visionHarris, open("visionHarris.pkl", "wb"))



if __name__ == "__main__":


    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()