from .functions import *


class OrthogonalForms:
    def __init__(self, function):
        self.function = function

    def eval_form(self, x):
        return (
            self.function.eval_poly(x) *
            math.sqrt(self._weight(x) / self._norm())
        )

    def _weight(self, x):
        raise NotImplementedError

    def _norm(self):
        raise NotImplementedError


class CharlierForm(OrthogonalForms):
    def _weight(self, x):
        alpha = self.function.params["alpha"]
        return math.exp(-alpha) * alpha ** x / math.factorial(x)

    def _norm(self):
        alpha = self.function.params["alpha"]
        n = self.function.order
        if n < 0:
            return 0
        else:
            return math.factorial(n) * alpha ** n
