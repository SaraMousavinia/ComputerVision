import numpy as np
import cv2
import random


def get_random_points (img, alpha):

    random.seed()

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------

    points_x = np.random.randint(img.shape[0], size=(alpha, 1))
    points_y = np.random.randint(img.shape[1], size=(alpha, 1))
    points = np.concatenate((points_x, points_y), axis=1)
    # ----------------------------------------------

    return points


# start of some code for testing get_random_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    points = get_random_points(img, 50)
    print(points)

