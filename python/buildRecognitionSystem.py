import pickle
import cv2
import numpy as np
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features

def getTrainFeatures(train_imagenames, method, dictionarySize):

    trainFeatures = np.zeros((len(train_imagenames), dictionarySize))
    for i in range(len(train_imagenames)):
        img_name = train_imagenames[i]

        # Load wordmap
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (img_name[:-4], method), 'rb'))

        # Get images features
        image_features = get_image_features(wordMap, dictionarySize)

        # Append
        trainFeatures[i] = image_features

    return trainFeatures


print("Building recognition system")

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']
train_labels = meta['train_labels']

dictionaryHarris = pickle.load(open("dictionaryHarris.pkl", "rb"))
dictionaryRandom = pickle.load(open("dictionaryRandom.pkl", "rb"))

# In each pickle store a dictionary that contains:
# 1. dictionary:  your visual word dictionary, a matrix of size K x 3n
# 2. filterBank:  filter bank used to produce the dictionary.  This is an array of image filters
# 3. trainFeatures:  T x K matrix containing all of the histograms of visual words of the T training images in the data set.
# 4. trainLabels:  T x 1 vector containing the labels of each training image.

visionRandom = {'dictionary': None, 'filterBank': None, 'trainFeatures': None, 'trainLabels': None}
visionHarris = {'dictionary': None, 'filterBank': None, 'trainFeatures': None, 'trainLabels': None}

# 1. dictionary:
visionRandom['dictionary'] = dictionaryRandom
visionHarris['dictionary'] = dictionaryHarris

# 2. filterBank:
fb = create_filterbank()
visionRandom['filterBank'] = fb
visionHarris['filterBank'] = fb

# 3. trainFeatues:
trainFeaturesRandom = getTrainFeatures(train_imagenames, 'Random', dictionaryRandom.shape[0])
trainFeaturesHarris = getTrainFeatures(train_imagenames, 'Harris', dictionaryHarris.shape[0])
visionRandom['trainFeatures'] = trainFeaturesHarris
visionHarris['trainFeatures'] = trainFeaturesRandom

# 4.  trainLabels:
visionRandom['trainLabels'] = train_labels
visionHarris['trainLabels'] = train_labels

# Save to pickle file
pickle.dump(visionRandom, open("visionRandom.pkl", "wb"))
pickle.dump(visionHarris, open("visionHarris.pkl", "wb"))

print("Done!")


if __name__ == "__main__":


    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()