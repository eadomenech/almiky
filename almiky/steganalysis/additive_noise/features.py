import imageio
import numpy as np
from pathlib import Path


class ProcessImageFolder:
    '''
    Extract features of images from a folder
    using self.extractor
    '''

    def __init__(self, extractor):
        self.extractor = extractor

    def _process(self, file):
        image = imageio.imread(str(file))
        value = self.extractor(image)
        return value

    def __call__(self, folder):
        return [
            self._process(file)
            for file in folder.iterdir()
        ]
