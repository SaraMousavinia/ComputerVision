import cv2
import numpy as np
from utils import *
from createFilterBank import create_filterbank


def extract_filter_responses (img, filterBank):

    if len(img.shape) == 2:
        img = cv2.merge ([img, img, img])

    img = cv2.cvtColor (img, cv2.COLOR_BGR2Lab)

    # -----fill in your implementation here --------



    # ----------------------------------------------

    return filterResponses

i# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank ()

    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")

#    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
#    print (extract_filter_responses (gray, fb))

    print (extract_filter_responses (img, fb))

