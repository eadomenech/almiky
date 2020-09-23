'''
Basic hiders
'''
import itertools

import numpy as np

from almiky.utils.blocks import BlocksImage


class BlockHider:
    '''
    Abstract class to hide payload in blocks.
    '''

    def __init__(self, hider):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.hider = hider

    def insert(self, cover, payload, block_shape=(8, 8)):
        '''
        Hide the payload in a cover work.

        Arguments:
        cover -- cover work
        payload -- payload
        block_shape: -- block dimensions
        '''
        raise NotImplementedError

    def extract(self, ws_work, block_shape=(8, 8)):
        '''
        Extract payload from watermarked/stego work

        Arguments:
        ws_cover -- watermarked/stego work
        block_shape: -- block dimensions
        '''
        raise NotImplementedError


class BlockBitHider(BlockHider):
    '''
    Hide a bit in one coefficient.

    Build an hider from a scanner and embedder:
    hider = SingleBitHider(scan, embeder)

    then you can insert a bit in a coefficient
    index = 0
    hider.insert(1, index)

    or extract a bit from a coefficient
    hider.extract(10)
    '''

    def __init__(self, hider):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.hider = hider

    def insert(self, cover, msg, block_shape=(8, 8), **kwargs):
        '''
        Hide a bit

        Arguments:
        bit -- bit to hide
        index -- index of coefficient where bit will be hidden
        '''
        data = np.copy(cover)
        blocks = BlocksImage(data, *block_shape)

        for i in range(len(msg)):
            try:
                blocks[i] = self.hider.insert(blocks[i], msg[i], **kwargs)
            except IndexError as e:
                raise ValueError("Capacity exceded.") from e

        return data

    def extract(self, ws_work, block_shape=(8, 8), **kwargs):
        '''
        Get bit hidden an return it

        Arguments:
        index -- index of coefficient where bit will be extracted
        '''
        msg = ''
        blocks = BlocksImage(ws_work, *block_shape)

        for block in blocks:
            bit = self.hider.extract(block, **kwargs)
            msg += str(bit)

        return msg
