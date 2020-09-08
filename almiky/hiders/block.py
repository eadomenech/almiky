'''
Basic hiders
'''
import itertools

from almiky.utils.blocks import BlocksImage


class BlockBitHider:
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

    def insert(self, cover, msg, index=0, block_shape=(8, 8)):
        '''
        Hide a bit

        Arguments:
        bit -- bit to hide
        index -- index of coefficient where bit will be hidden
        '''
        blocks = BlocksImage(cover, *block_shape)

        for block, bit in zip(blocks, msg):
            self.hider.insert(block, int(bit), index)

    def extract(self, ws_work, index=0, block_shape=(8, 8)):
        '''
        Get bit hidden an return it

        Arguments:
        index -- index of coefficient where bit will be extracted
        '''
        msg = ''
        blocks = BlocksImage(ws_work, *block_shape)

        for block in blocks:
            bit = self.hider.extract(block, index)
            msg += str(bit)

        return msg
