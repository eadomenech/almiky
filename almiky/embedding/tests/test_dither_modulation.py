'''
Embedding module tests
'''

import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from almiky.embedding.dither import DitherModulationQuantization


class QuantizationStepStrictlyPositive(unittest.TestCase):
    error_msg = "Quantization step must be strictly_positive"

    def test_quantization_step_equal_cero(self):
        with self.assertRaises(ValueError) as cm:
            DitherModulationQuantization(0)
        
        self.assertEqual(cm.exception.args[0], self.error_msg)

    def test_quantization_step_equal_less_than_cero(self):
        with self.assertRaises(ValueError) as cm:
            DitherModulationQuantization(-5)
        
        self.assertEqual(cm.exception.args[0], self.error_msg)


class EmbeddingTest(unittest.TestCase):

    def test_embedding_cero_possite_coefficient(self):
        embedder = DitherModulationQuantization(step=10)
        self.assertEqual(embedder.embed(coefficient=39.0, bit=0), 35)

    def test_embedding_cero_negative_coefficient(self):
        embedder = DitherModulationQuantization(step=10)
        self.assertEqual(embedder.embed(coefficient=-50.5, bit=0), 55)


if __name__ == '__main__':
    unittest.main()