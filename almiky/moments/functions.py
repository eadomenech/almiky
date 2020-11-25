# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal functions and provide operations for it evaluation.
Each orthogonal function is defined in a derivated class of OrtogonalFunction.
'''

import math
from scipy import special
from mpmath import qp, mp, qhyper, hyp3f2


class OrtogonalFunction:
    '''
    Abastract class that represent a polinomial function used to calculate
    ortogonal moments. This class do not represent any particular polinomial
    function; should be derivated for set the evaluator by implementing
    _eval(...) method.

    class FunctionX(OrtogonalFunction)
        def keval(...)
            ...

    FunctionX(order, \**parameters) => new orthogonal function with
    an specific order and parameters.

    For example: FunctionX(8, alpha=0.2, beta=0.3)
    '''

    def eval(self, x, order):
        '''
        func.eval(x) => double, return evaluation of the ortogonal function
        in x
        '''
        values = (self.keval(x, k, order) for k in range(order + 1))
        return sum(values)

    def keval(self, x, k, order):
        '''
        func.keval(a) => double, return evaluation of the ortogonal function
        in x for specific order
        '''
        raise NotImplementedError

    def norm(self, order):
        '''
        func.norm(x) => double, return the norm of orthogonal function
        '''
        raise NotImplementedError


class QHahnFunction(OrtogonalFunction):

    def __init__(self, q, alpha, beta, N):
        super().__init__()
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.N = N

    def keval(self, x, k, order):
        mp.dps = 25
        mp.pretty = True
        return (
            self.q ** k *
            qp(self.q ** -order, self.q, k) *
            qp(self.alpha * self.beta * self.q ** (order + 1), self.q, k) *
            qp(self.q ** -x, self.q, k) *
            qp(self.alpha * self.q, self.q, k) ** -1 *
            qp(self.q ** -self.N, self.q, k) ** -1 /
            qp(self.q, self.q, k)
        )

    def norm(self, order):
        if order < 0:
            return 0
        else:
            return (
                qp(self.alpha * self.beta * self.q ** 2, self.q, self.N) *
                qp(self.q, self.q, order) *
                qp(
                    self.alpha *
                    self.beta *
                    self.q ** (self.N + 2),
                    self.q,
                    order
                ) *
                qp(self.beta * self.q, self.q, order) *
                qp(self.beta * self.q, self.q, self.N) ** -1 *
                qp(self.alpha * self.q, self.q, order) ** -1 *
                qp(self.alpha * self.beta * self.q, self.q, order) ** -1 *
                qp(self.q ** -self.N, self.q, order) ** -1 *
                (self.alpha * self.q) ** -self.N *
                (1 - self.alpha * self.beta * self.q) *
                (-self.alpha * self.q) ** order *
                (1 - self.alpha * self.beta * self.q ** (2 * order + 1)) **
                -1 * self.q ** (special.binom(order, 2) - self.N * order)
            )


class CharlierFunction(OrtogonalFunction):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def keval(self, x, k, order):
        return (
            (-1) ** (order - k) *
            special.poch(-order, k) *
            special.poch(-x, k) *
            self.alpha ** (order - k) /
            math.factorial(k)
        )

    def norm(self, order):
        if order < 0:
            return 0
        else:
            return math.factorial(order) * self.alpha ** order


class CharlierSobolevFunction(CharlierFunction):

    def __init__(self, alpha, beta, gamma):
        super().__init__(alpha)
        self.beta = beta
        self.gamma = gamma

    def kernel(self, x, order):
        return sum(
            [
                (i * super(CharlierFunction, self).eval(x, i - 1)) ** 2 /
                self.norm(i) for i in range(order)
            ]
        )

    def An(self, x, order):
        if order <= 0:
            return 1
        else:
            num = (
                self.beta *
                order *
                super().eval(self.gamma, order - 1) *
                (
                    super().eval(self.gamma, order - 1) +
                    (x - self.gamma) *
                    (order - 1) *
                    super().eval(self.gamma, order - 2)
                )
            )
            den = (
                self.norm(order - 1) *
                (1 + self.beta * self.kernel(self.gamma, order)) *
                (x - self.gamma) *
                (x - self.gamma - 1)
            )
            return 1 - num / den

    def Bn(self, x, order):
        if order <= 0:
            return 0
        else:
            num = (
                self.beta *
                order *
                super().eval(self.gamma, order - 1) *
                (
                    super().eval(self.gamma, order) + (x - self.gamma) *
                    order *
                    super().eval(self.gamma, order - 1)
                )
            )
            den = (
                self.norm(order - 1) *
                (1 + self.beta * self.kernel(self.gamma, order)) *
                (x - self.gamma) *
                (x - self.gamma - 1)
            )
            return num / den

    def eval(self, x, order):
        '''
        func.eval(x) => double, return evaluation of the ortogonal function
        in x
        '''
        return (
            self.An(x, order) * super().eval(x, order) +
            self.Bn(x, order) * super().eval(x, order - 1)
        )


class QKrawtchoukFunction(OrtogonalFunction):

    def __init__(self, p, q, N):
        self.q = q
        self.p = p
        self.N = N

    def keval(self, x, k, order):
        mp.dps = 25
        mp.pretty = True
        return (
            self.q ** k *
            qp(self.q ** -order, self.q, k) *
            qp(self.q ** -x, self.q, k) *
            qp(-self.p * self.q ** order, self.q, k) *
            qp(self.q ** -self.N, self.q, k) ** -1 *
            qp(0, self.q, k) ** -1 *
            qp(self.q, self.q, k) ** -1
        )

    def norm(self, order):
        # TODO: Who is n?
        n = None
        if order < 0:
            return 0
        else:
            return (
                qp(self.q, self.q, order) *
                qp(-self.p * self.q ** (self.N + 1), self.q, order) *
                qp(-self.p, self.q, order) ** -1 *
                qp(self.q ** -self.N, self.q, order) ** -1 *
                (1 + self.p) / (1 + self.p * self.q ** (2 * order)) *
                qp(-self.p * self.q, self.q, self.N) *
                self.p ** -self.N * self.q ** -special.binom(self.N + 1, 2) *
                (-self.p * self.q ** -self.N) ** order *
                self.q ** order ** 2
            )
            return (
                qp(self.q, self.q, n) *
                qp(-self.p * self.q ** (self.N + 1), self.q, n) *
                qp(-self.p, self.q, n) ** -1 *
                qp(self.q ** -self.N, self.q, n) ** -1 *
                (1 + self.p) / (1 + self.p * self.q ** (2 * n)) *
                qp(-self.p * self.q, self.q, self.N) *
                self.p ** -self.N * self.q ** -special.binom(self.N + 1, 2) *
                (-self.p * self.q ** -self.N) ** n *
                self.q ** n ** 2
            )


class TchebichefFunction():
    def __init__(self, N):
        self.N = N

    def eval(self, x, order):
        mp.dps = 25
        mp.pretty = True
        return (
            special.poch(1 - self.N, order) *
            hyp3f2(-order, -x, 1 + order, 1, 1 - self.N, 1)
        )

    def norm(self, order):
        mp.dps = 25
        mp.pretty = True
        return (
            math.factorial(2 * order) *
            special.binom(self.N + order, 2 * order + 1)
        )


class QCharlierFunction():
    def __init__(self, a, q):
        self.a = a
        self.q = q

    def eval(self, x, order):
        mp.dps = 25
        mp.pretty = True
        L = [self.q ** -order, self.q ** -x]
        return qhyper(L, [0], self.q, -self.q ** (order + 1) / self.a)

    def norm(self, order):
        mp.dps = 25
        mp.pretty = True
        return (
            self.q ** -order *
            special.qp(-self.a, self.q) *
            special.qp(-self.a ** -1 * self.q, self.q, order) *
            special.qp(self.q, self.q, order)
        )
