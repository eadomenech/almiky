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

    #FIXME: Conversion from binary to str must works to utf8 str
    '''def test_utf8(self):
        binary = '11011101100001110111010110000111011000111001011111010'

        self.assertEqual(utils.char2bin('núñez'), binary)'''


class BinaryToStrTest(TestCase):
    '''
    Tests for conversion from binary str sequence to str
    '''

    def test_empty_string(self):
        self.assertEqual(utils.bin2char(''), '')

    def test_hello(self):
        binary = '11010001100101110110011011001101111'

        self.assertEqual(utils.bin2char(binary), 'hello')

    #FIXME: Conversion from binary to str must works to utf8 str
    '''def test_utf8(self):
        binary = '01101110110000111011101011000011101100010110010101111010'

        self.assertEqual(utils.bin2char(binary), 'núñez')'''


if __name__ == '__main__':
    unittest.main()