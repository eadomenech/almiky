'''Noise attacks'''


import numpy as np
import random


def salt_paper_noise(image, density, max_value=255):
    '''Applies Salt and Pepper noise to image.

    Arguments:
    data -- image data as numpy array
    density -- probability at the pixels are altered (value between 0 an 1)
    max_value -- miximun image values

    Return: altered image as numpy array
    '''

    values = [0, max_value]
    x, y = image.shape
    altered = np.copy(image)

    for i in range(x):
        for j in range(y):
            alpha = random.uniform(0, 1)
            if alpha < density:
                beta = random.randint(0, 1)
                altered[i, j] = values[beta]

    return altered


def gaussian_noise(image, density, mu=0, sigma=0.5, max_value=255):
    '''Applies Gaussian noise to image.

    Arguments:
    data -- image data as numpy array
    density -- probability at the pixels are altered (value between 0 an 1)
    mu -- mean
    sigma -- standar deviation
    max_value -- maximun image values

    Return: altered image as numpy array
    '''

    x, y = image.shape
    altered = np.copy(image)

    for i in range(x):
        for j in range(y):
            alpha = random.uniform(0, 1)
            if alpha < density:
                offset = random.gauss(mu, sigma)
                new_value = round(image[i, j] + offset)
                if new_value < 0: new_value = 0
                elif new_value > max_value: new_value = max_value
                altered[i, j] = new_value if new_value >= 0 else 0

    return altered
