'''Tests for scalar quatization'''

from unittest import TestCase

from almiky.quantization import scalar


class UniformQuantizationTest(TestCase):

    def test_extreme_value(self):
        quantize = scalar.UniformQuantizer(5)

        self.assertEqual(quantize(0), 0)
        self.assertEqual(quantize(5), 5)
        self.assertEqual(quantize(10), 10)
        self.assertEqual(quantize(-5), -5)
        self.assertEqual(quantize(-10), -10)

    def test_intermediate_value(self):
        quantize = scalar.UniformQuantizer(5)

        self.assertEqual(quantize(5.5), 5)
        self.assertEqual(quantize(5.8), 5)
        self.assertEqual(quantize(1.5), 0)
        self.assertEqual(quantize(2.4), 0)
        self.assertEqual(quantize(7.6), 10)
        self.assertEqual(quantize(-8.2), -10)
