from unittest import TestCase
from unittest.mock import Mock

from almiky.utils.optimization.functions import multiple
from almiky.utils.functions import WeightedFunction


class ConventionalWeightedAggregationTest(TestCase):
    def test_evaluation(self):
        f = Mock(side_effect=lambda x: x * 2)
        g = Mock(side_effect=lambda x: x * 3)
        wf = WeightedFunction(f, 0.2)
        wg = WeightedFunction(g, 0.8)

        F = multiple.WeightedAggretation((wf, wg))
        value = F(5)

        f.assert_called_once()
        g.assert_called_once()

        self.assertEqual(value, 14)

    def test_functions_called_with_same_arguments(self):
        '''
        Test that an exception is raised when no
        functions is passed.
        '''

        with self.assertRaises(ValueError):
            multiple.WeightedAggretation([])

    def test_evaluation_without_functions(self):
        '''
        Test that an exception is raised when no
        functions is passed.
        '''
        f = Mock(return_value=1)
        g = Mock(return_value=1)
        wf = multiple.WeightedAggretation((f, g))
        wf(5, 6, a=7, b=8)

        f.assert_called_once_with(5, 6, a=7, b=8)
        g.assert_called_once_with(5, 6, a=7, b=8)
