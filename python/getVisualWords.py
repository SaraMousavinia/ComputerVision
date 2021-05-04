import pickle

import numpy as np
from scipy.spatial.distance import cdist
import cv2
from skimage.color import label2rgb

from python.createFilterBank import create_filterbank
from python.extractFilterResponses import extract_filter_responses


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



    # for i in range(100):
    #     temp = dictionary[i]
    #     for j in range(60):
    #
    #
    #
    #         pixelValue = filterResponses[i] #[x][y]
    #         # pred = dictionary.predict(pixelValue)
    #         print(pixelValue.shape)
    #         # result = cdist(filterResponses[j], dictionary)
    #         print(result)
            # result = cdist(dictionary, pixelValue)

        # result = cdist(dictionary, filterResponses[i])
        # print(result.shape,  "resultshape")
        # wordMap.append(result)

    print(wordMap.shape)
    # Y = cdist(None, None, metric='euclidean')


    # ----------------------------------------------

    return wordMap


if __name__ == '__main__':

    fb = create_filterbank()
    img = cv2.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")

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