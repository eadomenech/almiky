'''Embedding methods based on quantization index modulation (QIM)'''


class BaseBinaryQIM:
    '''Quantization index modulation is used to embed one bit'''
    def __init__(self, step):
        '''Init class attr

        Arguments:
        step -- step size used for embedding and extraction.
        '''
        self.step = step


class BinaryQuantizationIndexModulation(BaseBinaryQIM):
    '''Quantization index modulation is used to embed one bit'''

    def embed(self, amplitude, bit):
        '''
        Embedds a bit and return the new amplitude value.
        Arguments:
        amplitude -- amplitude of signal
        bit -- bit to embedd
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
        Arguments:
        amplitude -- amplitude of signal
        bit -- bit to embedd
        '''
        amplitude_diffs = list(map(
            lambda bit: abs(self.embed(amplitude, bit) - amplitude),
            (0, 1)
        ))
        return amplitude_diffs.index(min(amplitude_diffs))


class BinaryQIM(BaseBinaryQIM):
    '''
    Quantization index modulation is used to embed one bit.
    From: Chen, B., & Wornell, G. W. (2001). Quantization index modulation: 
    A class of provably good methods for digital watermarking and
    information embedding. IEEE Transactions on
    Information theory, 47(4), 1423-1443.
    With: d[k,0] = -∆/4
    '''

    def embed(self, amplitude, bit):
        '''
        Embedds a bit and return the new amplitude value.
        Arguments:
        amplitude -- amplitude of signal
        bit -- bit to embedd
        '''
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError('Embedding an invalid bit')

        if not bit:
            new_amplitude = (
                self.step *
                round(amplitude / self.step + 0.25) -
                self.step / 4.0)
        else:
            new_amplitude = (
                self.step *
                round(amplitude / self.step - 0.25) +
                self.step / 4.0)

        return new_amplitude

    def extract(self, amplitude):
        '''Extracts the embedded bit according to the amplitude value.
        Arguments:
        amplitude -- amplitude of signal
        bit -- bit to embedd
        '''
        one = self.embed(amplitude, 1)
        zero = self.embed(amplitude, 0)
        if abs(amplitude - one) < abs(amplitude - zero):
            bit = 1
        else:
            bit = 0

        return bit