import numpy as np
import cv2
from python.extractFilterResponses import extract_filter_responses
from python.getRandomPoints import get_random_points
from python.getHarrisPoints import get_harris_points
from python.createFilterBank import create_filterbank

from sklearn.cluster import KMeans

CORNER_HARRIS = 'Harris'
RANDOM = 'Random'

def get_dictionary(imgPaths, alpha, K, method=CORNER_HARRIS):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv2.imread ('../data/%s' % path)

        # TODO: Use RGB instead of BGR?
        # should be OK in standard BGR format
        # image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        
        # -----fill in your implementation here --------

        filterResponses = extract_filter_responses(image, filterBank)

        if method == CORNER_HARRIS:
            get_harris_points(filterResponses, alpha)
        elif method == RANDOM:
            get_random_points(filterResponses, alpha)
        else:
            raise Exception("Incompatible method")








        # ----------------------------------------------

    # can use either of these K-Means approaches...  (i.e. delete the other one)
    # OpenCV K-Means
    # pixelResponses = np.float32 (pixelResponses)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret,label,dictionary=cv2.kmeans(pixelResponses,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # sklearn K-Means
    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary

if __name__ == "__main__":
    img1_path = "bedroom/sun_aiydcpbgjhphuafw.jpg"
    img2_path = "desert/sun_adpbjcrpyetqykvt.jpg"

    imgPaths = [img1_path, img2_path]
    K = 100
    method = None

    result = get_dictionary(imgPaths, 50, K, method)

    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()
