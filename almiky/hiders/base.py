'''
Basic hiders
'''
import numpy as np


class SingleBitHider:
    '''Hide a bit in one coefficient.

    Build an hider from a scanner and embedder:
        hider = SingleBitHider(scan, embedder)

    then you can insert a bit in a coefficient:
        index = 0 \n
        hider.insert(1, index)

    or extract a bit from a coefficient:
        hider.extract(10)

    Attributes:
        scan (ScanMapping):
        embedder (Embedder):
    '''

    def __init__(self, scan, embedder):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.embedder = embedder
        self.scan = scan

    def insert(self, cover_work, bit, index=0):
        '''
        Hide a bit

        Args:
            cover_work (numpy array): cover Work array
            bit (int): bit to hide
            index (int): index of coefficient where bit will be hidden
        '''
        data = np.copy(cover_work)
        scanning = self.scan(data)
        amplitude = scanning[index]
        scanning[index] = self.embedder.embed(amplitude, bit)

        return data

    def extract(self, ws_work, index=0):
        '''
        Get bit hidden an return it

        Args:
            ws_work (numpy array): watermarked or stego Work array
            index (int): index of coefficient where bit will be extracted
                (default is 0)
        '''
        scanning = self.scan(ws_work)
        amplitude = scanning[index]
        return self.embedder.extract(amplitude)


class TransformHider:
    '''
    Gives to an arbitrary hider the capacity
    to hide payload in transform domain using
    an arbitrary transform too.

    hider and transform dependencies are set in itialization:
        hider = TransformHider(base_hider, transform)

    This class implement hider interface:
        hider.insert(cover_work, ...) \n
        hider.extract(ws_work, ....)

    Aditional arguments are pased to based hider.

    Attributes:
        hider:
        transform:
    '''
    def __init__(self, hider, transform):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.hider = hider
        self.transform = transform

    def insert(self, cover_work, data, **kwargs):
        '''
        Insert the payload in transform domain using
        base hider.
        '''
        direct = self.transform.direct(cover_work)
        ws_work = self.hider.insert(direct, data, **kwargs)

        return self.transform.inverse(ws_work)

    def extract(self, ws_work, **kwargs):
        '''
        Extract payload from transform domain using
        base hider.
        '''
        direct = self.transform.direct(ws_work)

        return self.hider.extract(direct, **kwargs)
