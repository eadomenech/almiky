'''
Tests for utility module
'''

import unittest
from unittest import TestCase

from almiky.utils import utils


class StrToBinaryTest(TestCase):
    '''
    Tests for conversion from str to binary str sequence
    '''

    def test_empty_string(self):
        self.assertEqual(utils.char2bin(''), '')

    def test_hello(self):
        binary = '11010001100101110110011011001101111'

        self.assertEqual(utils.char2bin('hello'), binary)

    @unittest.expectedFailure
    def test_utf8(self):
        binary = '11011101100001110111010110000111011000111001011111010'

        self.assertEqual(utils.char2bin('núñez'), binary)


class BinaryToStrTest(TestCase):
    '''
    Tests for conversion from binary str sequence to str
    '''

    def test_empty_string(self):
        self.assertEqual(utils.bin2char(''), '')

    def test_hello(self):
        binary = '11010001100101110110011011001101111'

        self.assertEqual(utils.bin2char(binary), 'hello')

    @unittest.expectedFailure
    def test_utf8(self):
        binary = '01101110110000111011101011000011101100010110010101111010'

        self.assertEqual(utils.bin2char(binary), 'núñez')


class MaxPSNRTest(TestCase):
    def test_default_amplitude(self):
        '''
        Testing with aplitude equal to 255
        '''

        dimensions = (256, 256)
        expected = 96.295602915
        value = utils.max_psnr(dimensions)

        self.assertAlmostEqual(value, expected)

    def test_custom_amplitude(self):

        dimensions = (256, 256)
        expected = 102.350198526
        value = utils.max_psnr(dimensions, max=512)

        self.assertAlmostEqual(value, expected)


if __name__ == '__main__':
    unittest.main()