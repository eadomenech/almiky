'''Scalar quantization module'''


class UniformQuantizer:
    '''
    Scalar uniform quantizer

    Arguments:
    step: quantization step size (∆)

    UniformQuantizer(10) => new uniform quantizer with ∆=10

    q = UniformQuatizer(10)
    q(10) => quatization of value 10
    '''

    def __init__(self, step):
        '''Initialize x; see help(type(x)) for details'''
        self.step = step

    def __call__(self, amplitude):
        return self.step * round(amplitude / self.step)
