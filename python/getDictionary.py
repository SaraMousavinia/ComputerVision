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
    row_counter = 0
    print(pixelResponses.shape)
    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv2.imread('../data/%s' % path)

        # TODO: Use RGB instead of BGR?
        # should be OK in standard BGR format
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        
        # -----fill in your implementation here --------

        filterResponses = extract_filter_responses(image, filterBank)

        if method == CORNER_HARRIS:
            points = [get_harris_points(filterResponses[i], alpha) for i in range(60)]
        elif method == RANDOM:
            points = [get_random_points(filterResponses[i], alpha) for i in range(60)]
        else:
            raise Exception("Incompatible method")

        # For each row in pixelResponses we first get
        # Points contains the coordinates of the value in the filterResponses that we want to put in the pixelResponses
        for j in range(alpha):
            for k in range(len(points)):
                point_x = points[k][j][0]
                point_y = points[k][j][1]
                pixelResponses[row_counter][k] = filterResponses[k][point_x, point_y]
            row_counter += 1

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
    method = CORNER_HARRIS

    dictionary = get_dictionary(imgPaths, 50, K, method)
    print(dictionary.shape)
    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()
