'''
Robustness performance metrics
'''


def ber(iwatermark, ewatermark):
    '''
    Calculate a return Bit Error Rate (BER). Raise an exception
    if almost one of watermarks is empty or are invalid binary
    sequence. Is posible to process watermarks of different size,
    min size bits are only tested.

    Arguments:
    iwatermark -- watermark inserted as str of binary data
    iwatermark -- watermark inserted as str of binary data

    Usage:
    ber('0101110', '0111011111')
    ber('0101110', '0111011')
    ber('', '10110') => raise ValueError
    ber('1011', '') => raise ValueError
    ber('1011', '') => raise ValueError
    '''

    def check(ibit, ebit):
        '''
        Check if two bit are equals; return 1 if are not equals,
        return 0 otherwise. Raise an exception if arguments are
        invalid bits.

        Arguments:
        ibit -- bit represented as str object
        ebit -- bit represented as str object
        '''
        if any((bit not in ('0', '1') for bit in (ibit, ebit))):
            raise ValueError

        return int(ibit) ^ int(ebit)

    try:
        incorrectly_decoded = sum(
            (
                check(ibit, ebit)
                for ibit, ebit in
                zip(iwatermark, ewatermark)
            )
        )
        number_bits = min(len(iwatermark), len(ewatermark))

        return incorrectly_decoded / number_bits

    except ZeroDivisionError:
        raise ValueError('Watermark length must not be empty')
    except ValueError:
        raise ValueError('Invalid binary data')
