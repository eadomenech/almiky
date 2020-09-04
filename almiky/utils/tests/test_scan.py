'''
Tests for utility module
'''

import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock

import numpy as np

from almiky.utils.scan.scan import ScanMapping
from almiky.utils.scan import maps


class DefaultMappingScanTest(TestCase):
    '''
    Test scanning with default map
    '''

    def test_scanning(self):
        '''
        Test scanning with 8x8 row major scan
        '''
        block = np.array(range(0, 64)).reshape(8, 8)

        exploration = list(range(0, 64))

        scanning = ScanMapping(block)

        np.testing.assert_array_equal(list(scanning), exploration)

    def test_get_coeficient(self):
        '''
        Test get coefficient with 8x8 row major scan
        '''
        block = np.array(range(0, 64)).reshape(8, 8)

        scanning = ScanMapping(block)

        self.assertEqual(scanning[0], 0)
        self.assertEqual(scanning[18], 18)
        self.assertEqual(scanning[63], 63)

    def test_map_index_out_range(self):

        block = np.array(range(0, 64)).reshape(8, 8)

        scanning = ScanMapping(block)

        with self.assertRaises(IndexError, msg='scan index out of range'):
            scanning[64]

        with self.assertRaises(IndexError, msg='scan index out of range'):
            scanning[64] = 9

    def test_block_index_out_range(self):
        '''
        Test raise condition when block has
        less dimension than the map
        '''
        block = np.array([
            [0, 1, 5 , 6,],
            [2, 4, 7, 13],
            [3, 8, 12, 17],
            [9, 11, 18, 24]
        ])

        scanning = ScanMapping(block)

        with self.assertRaises(IndexError, msg='block index out of range'):
            scanning[63]

        with self.assertRaises(IndexError, msg='block index out of range'):
            scanning[63] = 8

    def test_set_coeficient(self):
        '''
        Test set coefficient with 8x8 row major scan
        '''
        block = np.array(range(0, 64)).reshape(8, 8)

        scanning = ScanMapping(block)
        scanning[24] = 8

        self.assertEqual(block[3, 0], 8)


class MappingScanTest(TestCase):
    '''
    Test scanning with custom map
    '''

    def test_scanning(self):
        '''
        Test scanning with 8x8 zig-zag scan
        '''
        block = np.array([
            [0, 1, 5 , 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29 , 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34 , 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])

        exploration = list(range(0, 64))

        scanning = ScanMapping(block, maps.ZIGZAG_8x8)
        
        np.testing.assert_array_equal(list(scanning), exploration)

    def test_get_coeficient(self):
        '''
        Test get coefficient with 8x8 zig-zag scan
        '''
        block = np.array([
            [0, 1, 5 , 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29 , 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34 , 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])

        scanning = ScanMapping(block, maps.ZIGZAG_8x8)

        self.assertEqual(scanning[0], 0)
        self.assertEqual(scanning[18], 18)
        self.assertEqual(scanning[63], 63)

    def test_map_index_out_range(self):

        block = np.array([
            [0, 1, 5 , 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29 , 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34 , 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])

        scanning = ScanMapping(block, maps.ZIGZAG_8x8)

        with self.assertRaises(IndexError, msg='scan index out of range'):
            scanning[64]

        with self.assertRaises(IndexError, msg='scan index out of range'):
            scanning[64] = 9

    def test_block_index_out_range(self):
        '''
        Test raise condition when block has
        less dimension than the map
        '''
        block = np.array([
            [0, 1, 5 , 6,],
            [2, 4, 7, 13],
            [3, 8, 12, 17],
            [9, 11, 18, 24]
        ])

        scanning = ScanMapping(block, maps.ZIGZAG_8x8)

        with self.assertRaises(IndexError, msg='block index out of range'):
            scanning[63]

        with self.assertRaises(IndexError, msg='block index out of range'):
            scanning[63] = 8

    def test_set_coeficient(self):
        '''
        Test set coefficient with 8x8 zig-zag scan
        '''
        block = np.array([
            [0, 1, 5 , 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29 , 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34 , 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])

        scanning = ScanMapping(block, maps.ZIGZAG_8x8)
        scanning[24] = 8

        self.assertEqual(block[3, 3], 8)
