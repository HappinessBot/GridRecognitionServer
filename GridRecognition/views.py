from django.http import JsonResponse

import cv2
import numpy as np
import math

SIZE_OF_GRID = 11
FIX_CENTERS = 2

IMAGE_WIDHT = 13
IMAGE_HEIGHT = 13
N_MIN_ACTVE_PIXELS = 10


# Create your views here.
def get_matrix(request):
    image = get_image()
    centers = get_centers(image)
    matrix = make_matrix(image, centers)
    return JsonResponse({'matrix': matrix})


def get_image():
    image = cv2.imread('pics/2017-07-12-165523.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 29, 15)

    thresh2, contours0, hierarchy = cv2.findContours(thresh,
                                                     cv2.RETR_LIST,
                                                     cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]

    size_rectangle_max = 0

    squares = []

    big_rectangle = []

    for i in range(len(contours0)):
        approximation = cv2.approxPolyDP(contours0[i], 4, True)

        if(not (len(approximation) == 4)):
            continue

        if(not cv2.isContourConvex(approximation)):
            continue

        size_rectangle = cv2.contourArea(approximation)

        squares.append(approximation)

        if size_rectangle > size_rectangle_max:
            size_rectangle_max = size_rectangle
            big_rectangle = approximation

    def getOuterPoints(rcCorners):
        ar = []
        ar.append(rcCorners[0, 0, :])
        ar.append(rcCorners[1, 0, :])
        ar.append(rcCorners[2, 0, :])
        ar.append(rcCorners[3, 0, :])

        x_sum = sum(rcCorners[x, 0, 0]
                    for x in range(len(rcCorners))) / len(rcCorners)
        y_sum = sum(rcCorners[x, 0, 1]
                    for x in range(len(rcCorners))) / len(rcCorners)

        def algo(v):
            return (math.atan2(v[0] - x_sum, v[1] - y_sum) +
                    2 * math.pi) % 2 * math.pi
            ar.sort(key=algo)
        return (ar[3], ar[0], ar[1], ar[2])

    points1 = np.array([
                        np.array([0.0, 0.0], np.float32) + np.array([144, 0], np.float32),
                        np.array([0.0, 0.0], np.float32),
                        np.array([0.0, 0.0], np.float32) + np.array([0.0, 144], np.float32),
                        np.array([0.0, 0.0], np.float32) + np.array([144, 144], np.float32),
                        ], np.float32)
    outerPoints = getOuterPoints(big_rectangle)
    points2 = np.array(outerPoints, np.float32)

    pers = cv2.getPerspectiveTransform(points2, points1)

    warp = cv2.warpPerspective(image, pers,
                               (SIZE_OF_GRID * IMAGE_HEIGHT,
                                SIZE_OF_GRID * IMAGE_WIDHT))

    return warp


def get_centers(image):
    height, width, channels = image.shape
    centers = find_centers_of_boxes(height, width)
    centers = [centers[i:i + SIZE_OF_GRID] for i in range(0,
                                                          len(centers),
                                                          SIZE_OF_GRID)]
    return centers


def find_centers_of_boxes(height, width):
    centers = []
    for x in range(int(height / (2 * SIZE_OF_GRID)) + FIX_CENTERS,
                   height, int(height / SIZE_OF_GRID)):
        for y in range(int(width / (2 * SIZE_OF_GRID)) + FIX_CENTERS,
                       width, int(width / SIZE_OF_GRID)):
            centers.append((x, y))
    return centers


def make_matrix(image, centers):
    result = []
    for row in centers:
        result_row = []
        for el in row:
            if is_white(image[el[0]][el[1]]):
                result_row.append(0)
            else:
                result_row.append(1)
        result.append(result_row)
    return result


def is_white(center):
    return center[0] > 150 and center[1] > 150 and center[2] > 150
