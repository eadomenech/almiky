'''Methods for scanning'''


import numpy as np


class ScanMapping:
    def __init__(self, block, map):
        '''
        Initialization

        Arguments:
        block -- block to scan
        map -- map used for scanning
        '''
        self.block = block
        self.map = map
        self.length = self.block.shape[1]

    def _get_indexes(self, pos):
        '''
        Return an coefficient

        Arguments:
        pos -- map index
        '''
        x = int(pos / self.length)
        y = pos % self.length

        if x >= self.length:
            raise IndexError('block indexes out of range')

        return x, y

    def get_pos(self, index):
        '''
        Return coefficient position in block

        Arguments:
        index -- map index
        '''
        try:
            pos = self.map[index]
        except IndexError as e:
            raise IndexError('scan index out of range') from e

        return pos

    def __iter__(self):
        for i in self.map:
            x, y = self._get_indexes(i)
            yield self.block[x, y]

    def __getitem__(self, index):
        '''
        Return a coeficient

        Arguments:
        index -- coefficient index
        '''
        pos = self.get_pos(index)
        x, y = self._get_indexes(pos)

        return self.block[x, y]

    def __setitem__(self, index, value):
        '''
        Return a coeficient

        Arguments:
        index -- coefficient index
        '''
        pos = self.get_pos(index)
        x, y = self._get_indexes(pos)

        self.block[x, y] = value


ScanMapping.ZIGZAG_8x8 = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]
