import cv2
import numpy as np


def get_image_features (wordMap, dictionarySize):

    # -----fill in your implementation here --------

    h = None



    # ----------------------------------------------
    
    return h


if __name__ == "__main__":


    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()