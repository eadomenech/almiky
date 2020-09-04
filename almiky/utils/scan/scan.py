'''Methods for scanning'''


import numpy as np

from .maps import ROW_MAJOR_8x8


class ScanMapping:
    def __init__(self, block, map=ROW_MAJOR_8x8):
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
