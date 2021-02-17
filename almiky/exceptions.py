# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


class RegisterNotFound(Exception):

    def __init__(self):
        msg = "Register does not exist."
        super().__init__(msg)


class ExceededCapacity(Exception):

    def __init__(self):
        msg = "The message length exceeds the embedding capacity."
        super().__init__(msg)


class NotMatrixQuasiOrthogonal(Exception):

    def __init__(self):
        msg = "This matrix in not quasi orthogonal"
        super().__init__(msg)
