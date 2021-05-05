import pickle
import cv2
import numpy as np
from scipy.spatial import distance

from utils import chi2dist

EUCLIDEAN_DISTANCE = 'euclidean'
CHI2_DISTANCE = 'chi2'

def get_image_distance(hist1, hist2, method=EUCLIDEAN_DISTANCE):

    if method == EUCLIDEAN_DISTANCE:
        dist = distance.euclidean(hist1, hist2)
    elif method == CHI2_DISTANCE:
        dist = chi2dist(hist1, hist2)
    else:
        raise Exception("Incompatible distance method")

    return dist



if __name__ == "__main__":

    visionHarris = pickle.load(open("visionHarris.pkl", "rb"))
    visionRandom = pickle.load(open("visionRandom.pkl", "rb"))

    hist1 = visionHarris['trainFeatures'][0]
    hist2 = visionHarris['trainFeatures'][1]

    method = EUCLIDEAN_DISTANCE
    # method = CHI2_DISTANCE

    dist = get_image_distance (hist1, hist2, method)

    print(dist)


    # while True:
    #     k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
    #     if k in [27, 32]: break
    # cv2.destroyAllWindows()