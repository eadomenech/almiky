'''
Test for imperceptibility performance metrics
'''

import unittest

from almiky.metrics import robustness as metrics


class TestBER(unittest.TestCase):
    '''
    Test for Bit Error Rate (BER)
    '''
    def test_watermark_zero_length(self):
        # Watermark inserted
        iwatermark = ewatermark = ''

        with self.assertRaises(ValueError):
            metrics.ber(iwatermark, ewatermark)

    def test_invalid_binary_data(self):
        # Watermark inserted
        iwatermark = ewatermark = 'asd45621'

        with self.assertRaises(ValueError):
            metrics.ber(iwatermark, ewatermark)

        iwatermark = ewatermark = '45621'

        with self.assertRaises(ValueError):
            metrics.ber(iwatermark, ewatermark)

    def test_watermarks_different_length(self):
        # Watermark inserted
        iwatermark = '10110011'
        # Watermark extracted
        ewatermark = '1111000111'

        ber = metrics.ber(iwatermark, ewatermark)
        self.assertEqual(ber, 0.25)

    def test_watermarks_same_length(self):
        # Watermark inserted
        iwatermark = '1011001010'
        # Watermark extracted
        ewatermark = '1111000111'

        ber = metrics.ber(iwatermark, ewatermark)
        self.assertEqual(ber, 0.4)


if __name__ == '__main__':
    unittest.main()
