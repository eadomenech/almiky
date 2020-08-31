'''
Embedding module
'''


class DitherModulationQuantization:
    '''
    Dither modulation quantization technique.
    '''

    def __init__(self, step):
        '''
        Embedder initialization.

        Arguments
        step - quantization step size
        '''

        if step <= 0:
            raise ValueError("Quantization step must be strictly_positive", 'pepe')

        self.step = step

    def embed(self, coefficient, bit):
        '''
        Modify the value of cofficient to embed a bit, return modified coeficient.

        Arguments:
        coefficient - pixel value or transform coefficient
        bit - cero or one
        '''

        return (
            2 * self.step *
            round(abs(coefficient) / (2 * self.step)) -
            (self.step / 2)
        )
