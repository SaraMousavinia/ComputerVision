import pickle

import numpy as np
from scipy.spatial.distance import cdist
import cv2
from skimage.color import label2rgb

from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses


def get_visual_words(img, dictionary, filterBank):
    # -----fill in your implementation here --------
    wordMap = []

    filterResponses = extract_filter_responses(img, filterBank)

    wordMap = np.zeros((filterResponses.shape[1:]))
    # print(filterResponses.shape)
    # print(dictionary.shape)

    # result = cdist(filterResponses[0], dictionary)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            res = filterResponses[:, x, y]
            result = cdist([res], dictionary)
            # Add smallest distance
            wordMap[x, y] = result.min()

    # ----------------------------------------------

    return wordMap


if __name__ == '__main__':

    fb = create_filterbank()
    img = cv2.imread("../data/airport/sun_aesovualhburmfhn.jpg")

    cv2.imshow("Original", img)

    # Harris Dictionary
    dictionary = pickle.load(open("dictionaryHarris.pkl", 'rb'))
    wordMap = get_visual_words(img, dictionary, fb)
    result = label2rgb(wordMap)

    result = cv2.cvtColor(np.float32(result), cv2.COLOR_RGB2BGR)
    cv2.imshow("Result Harris", result)


    # Random Dictionary
    dictionary = pickle.load(open("dictionaryRandom.pkl", 'rb'))
    wordMap = get_visual_words(img, dictionary, fb)
    result = label2rgb(wordMap)

    result = cv2.cvtColor(np.float32(result), cv2.COLOR_RGB2BGR)
    cv2.imshow("Result Random", result)


    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()