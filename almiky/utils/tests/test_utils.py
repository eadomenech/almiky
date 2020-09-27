'''
Tests for utility module
'''

import unittest
from unittest import TestCase

from almiky.utils import utils


class BinaryConversion(TestCase):
    '''
    Tests for conversion from str to binary str sequence
    '''
    def test_empty_string(self):
        self.assertEqual(utils.bin2char(utils.char2bin('')), '')

    def test_ascii(self):
        msg = 'hello world'
        self.assertEqual(utils.bin2char(utils.char2bin(msg)), msg)

    def test_utf8(self):
        msg = 'díaz núñez'
        self.assertEqual(utils.bin2char(utils.char2bin(msg)), msg)


class MaxPSNRTest(TestCase):
    def test_default_max_amplitude(self):
        '''
        Evaluate with max aplitude equal to 255
        '''
        shape = (256, 256)
        expected = 96.295602915

        value = utils.max_psnr(shape)

        self.assertAlmostEqual(value, expected)

    def test_default_custom_amplitude(self):
        shape = (256, 256)
        expected = 102.350198526

        value = utils.max_psnr(shape, max=512)

        self.assertAlmostEqual(value, expected)


if __name__ == '__main__':
    unittest.main()
