'''Embedding methods based on quantization index modulation (QIM)'''


from almiky.embedding.base import Embedder


class BinaryQuantizationIndexModulation(Embedder):
    '''Quantization index modulation is used to embed one bit
    
    Attributes:
        step: step size used for embedding and extraction.
    '''

    def __init__(self, step):
        '''Init class attr.'''
        self.step = step

    def embed(self, amplitude, bit):
        '''Embedds a bit and return the new amplitude value.
        
        Args:
            amplitude (int): amplitude of signal
            bit (int): bit to embedd (0 or 1)
        '''
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError('Embedding an invalid bit')

        return (
            2 * self.step *
            round(amplitude / (2 * self.step)) +
            (bit or -1) / 2 * self.step
        )

    def extract(self, amplitude):
        '''Extracts the embedded bit according to the amplitude value.
        
        Args:
            amplitude (int): amplitude of signal
        
        Returns:
            int: embedded bit (0 or 1)
        '''
        amplitude_diffs = list(map(
            lambda bit: abs(self.embed(amplitude, bit) - amplitude),
            (0, 1)
        ))
        return amplitude_diffs.index(min(amplitude_diffs))
