# -*- coding: utf-8 -*-
import hashlib
import math

import numpy as np


def blake2b_bin(key):
    hexa_data = hashlib.blake2b(key.encode('utf-8')).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)


def replace(byte_init, bit):
    if bit == '0':
        if byte_init % 2 == 0:
            byte_fin = byte_init
        else:
            byte_fin = byte_init - 1
    elif bit == '1':
        if byte_init % 2 == 0:
            byte_fin = byte_init + 1
        else:
            byte_fin = byte_init
    return byte_fin


def ext_lsb(byte):
    if byte % 2 == 0:
        return '0'
    return '1'


def list2str(L):
    return "".join(str(i) for i in L)


def base_change(entrada, base, n=None):
    r = []
    pe = 1
    while pe != 0:
        pe = entrada // base
        r.insert(0, entrada % base)
        entrada = pe
    if n == None:
        return list2str(r)
    elif len(r) == n:
        return list2str(r)
    elif len(r) > n:
        return list2str(r[-n:])
    else:
        m = n - len(r)
        for i in range(m):
            r.insert(0, 0)
        return list2str(r)


def or_operation(x, y):
    if len(x) != len(y):
        raise ValueError("It is not possible to perform this operation")
    n = len(x)
    return "".join(("0" if x[i] == y[i] else "1") for i in range(n))


def increase(key, n):
    seq = blake2b_bin(key)
    while len(seq) < n:
        seq += blake2b_bin(seq[-128:])
    return seq[:n]


def bin2dec(bin_seq):
    dec_repr = 0
    for i in range(len(bin_seq)):
        if bin_seq[len(bin_seq) - i - 1] != '0':
            dec_repr += 2 ** i
    return dec_repr


def char2bin(data):
    return "".join(format(ord(x), '0b') for x in data)


def bin2char(bin_seq):
    return ''.join(
        (chr(int(bin_seq[i:i+7], 2)) for i in range(0, len(bin_seq), 7))
    )


def list_pos(cad):
    return [i for i in range(len(cad)) if cad[i] == "1"]


def permutation(cad):
    L = [i for i in range(len(cad)) if cad[i] == "1"]
    L.extend([i for i in range(len(cad)) if cad[i] == "0"])
    return L


def bitxor(bit_1, bit_2):
    if bit_1 == bit_2:
        return "0"
    else:
        return "1"


def lsb(byte):
    if byte % 2 == 0:
        return "0"
    else:
        return "1"


def matrix_zig_zag():
    A = np.array([
        [0,    1,     8,    16,     9,     2,     3,    10],
        [17,    24,    32,    25,    18,    11,     4,     5],
        [12,    19,    26,    33,    40,    48,    41,    34],
        [27,    20,    13,     6,     7,    14,    21,    28],
        [35,    42,    49,    56,    57,    50,    43,    36],
        [29,    22,    15,    23,    30,    37,    44,    51],
        [58,    59,    52,    45,    38,    31,    39,    46],
        [53,    60,    61,    54,    47,    55,    62,    63]])
    return A


def vzig_zag_scan(A):
    L = []
    pos = matrix_zig_zag().reshape(-1)
    seq = A.reshape(-1)
    for i in range(len(pos)):
        L.append(seq[pos[i]])
    return L


def inv_vzig_zag_scan(A):
    return vzig_zag_scan(A.T)


def mzig_zag_scan(vect):
    L = np.zeros(len(vect))
    pos = vzig_zag_scan(np.array(range(64)).reshape(8, 8))
    for i in range(len(pos)):
        L[int(pos[i])] = vect[i]
    return L.reshape(8, 8)


def max_psnr(shape, max=255):
    '''
    Determinate max psnr value (MSE = 0)

    Arguments:
    shape -- tuple: image dimensions
    max -- max amplitude value
    '''
    x, y = shape

    return 10 * math.log10(math.pow(max, 2) * x * y)
