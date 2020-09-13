from unittest import TestCase
from unittest.mock import Mock


from almiky.utils import functions


class WeightedFunctionTest(TestCase):
    def test_evaluate(self):
        f = Mock(side_effect=lambda x: x * 2)

        wf = functions.WeightedFunction(f, 0.2)
        value = wf(5)

        f.assert_called_once_with(5)
        self.assertEqual(value, 2)

    def test_base_function_called_with_same_arguments(self):
        f = Mock(return_value=5)

        wf = functions.WeightedFunction(f, 0.2)
        wf(5, 6, a=7, b=8)

        f.assert_called_once_with(5, 6, a=7, b=8)
