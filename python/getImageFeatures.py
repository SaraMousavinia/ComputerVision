import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------

    # h = cv2.calcHist([np.int8(wordMap)], [0], None, [50], [0,256])
    # hist, bins = np.histogram(wordMap.ravel(), 50, [-1000000, 1000000])
    # plt.hist(wordMap.ravel(), dictionarySize, [wordMap.min(), wordMap.max()]); plt.show()

    h = np.histogram(wordMap.ravel(), dictionarySize, [wordMap.min(), wordMap.max()])[0]

    # L1 Normalize histogram, relative to the size of the image
    h_norm = np.linalg.norm(h, ord=1)
    h_normalized = h/h_norm
    h = h_normalized

    # ----------------------------------------------

    return h


if __name__ == "__main__":

    wordMap = pickle.load(open('../data/airport/sun_aerinlrdodkqnypz_Harris.pkl', 'rb'))
    dictionarySize = 100  # K = 100

    image_features = get_image_features(wordMap, dictionarySize)

    print(image_features)
    print(image_features.sum())
    print(len(image_features))
    print(image_features[0])

    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()