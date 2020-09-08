'''
Basic hiders
'''


class SingleBitHider:
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

    def __init__(self, scan, embedder):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.embedder = embedder
        self.scan = scan

    def insert(self, cover, bit=1, index=0):
        '''
        Hide a bit

        Arguments:
        bit -- bit to hide
        index -- index of coefficient where bit will be hidden
        '''
        scanning = self.scan(cover)
        amplitude = scanning[index]
        bit = bit
        scanning[index] = self.embedder.embed(amplitude, bit)

    def extract(self, cover, index=0):
        '''
        Get bit hidden an return it

        Arguments:
        index -- index of coefficient where bit will be extracted
        '''
        scanning = self.scan(cover)
        amplitude = scanning[index]
        return self.embedder.extract(amplitude)
