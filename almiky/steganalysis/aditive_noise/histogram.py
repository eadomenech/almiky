import cv2
import numpy as np

def color_histogram(image):
    '''
    color_histogram(image) => numpy array: return histogram of color image
    @image: np array representative of image

    Calculate histogram of image per channel (R, G, B)
    using opencv calHist function. Array returned has (3, N) shape where
    N is bean of histogram
    '''

    red_histogram = cv2.calcHist([image],[0],None,[256],[0,256]).reshape(-1)
    green_histogram = cv2.calcHist([image],[1],None,[256],[0,256]).reshape(-1)
    blue_histogram = cv2.calcHist([image],[2],None,[256],[0,256]).reshape(-1)

    return np.array([red_histogram, green_histogram, blue_histogram])

