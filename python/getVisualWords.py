import pickle

import numpy as np
from scipy.spatial.distance import cdist
import cv2
from python.createFilterBank import create_filterbank
from python.extractFilterResponses import extract_filter_responses


def get_visual_words(img, dictionary, filterBank):
    # -----fill in your implementation here --------
    wordMap = []

    filterResponses = extract_filter_responses(img, filterBank)

    wordMaps = np.zeros((filterResponses.shape))

    print(wordMaps.shape)
    print(filterResponses.shape)
    print(dictionary.shape)

    for filter in filterResponses:
        pass

    # Y = cdist(None, None, metric='euclidean')


    # ----------------------------------------------

    return wordMap


if __name__ == '__main__':
    fb = create_filterbank()

    img = cv2.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")
    dictionary = pickle.load(open("dictionaryHarris.pkl", 'rb'))

    wordMap = get_visual_words(img, dictionary, fb)