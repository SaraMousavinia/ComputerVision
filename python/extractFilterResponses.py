import cv2
from python.createFilterBank import create_filterbank

def extract_filter_responses(img, filterBank):

    # if len(img.shape) == 2:
    #     img = cv2.merge([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # -----fill in your implementation here --------

    img_l = img[:, :, 0]
    img_a = img[:, :, 1]
    img_b = img[:, :, 2]

    # for i in filterBank:
    # filterResponses = []  # {}
    # for i in filterBank:
    #     filterResponses.append(cv2.filter2D(img, -1, filterBank[i]))

    filterResponses_l = [cv2.filter2D(img_l, -1, fb) for fb in filterBank]
    filterResponses_a = [cv2.filter2D(img_a, -1, fb) for fb in filterBank]
    filterResponses_b = [cv2.filter2D(img_b, -1, fb) for fb in filterBank]

    # ----------------------------------------------

    return filterResponses_l, filterResponses_a, filterResponses_b

# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank()

    img = cv2.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")

#    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
#    print (extract_filter_responses (gray, fb))

    filterResponses = extract_filter_responses(img, fb)
    cv2.imshow("original", img)
    print(len(filterResponses[0]), "123")
    print(len(filterResponses[1]), "123")
    print(len(filterResponses[2]), "123")
    # for i in range(0, 20):
    #     cv2.imshow("result_" + str(i), filterResponses[2][i])

    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k in [27, 32]: break
    cv2.destroyAllWindows()

