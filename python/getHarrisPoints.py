import numpy as np
import cv2


def get_harris_points(img, alpha, k=0.04):

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------

    sobel_aperture = 3
    block_size = 2
    corner_harris = cv2.cornerHarris(img, block_size, sobel_aperture, k)

    indices = (-corner_harris).argpartition(alpha, axis=None)[:alpha]
    points_x, points_y = np.unravel_index(indices, corner_harris.shape)
    points = np.concatenate(([points_x], [points_y]), axis=0).T

    # ----------------------------------------------
    
    return points


# start of some code for testing get_harris_points()
if __name__ == "__main__":
    img = cv2.imread("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    points = get_harris_points(img, 50, 0.04)
    print(points.shape)
    print(points)

