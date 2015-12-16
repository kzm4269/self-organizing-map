import numpy as np
import itertools

_XY2HEX = np.array([[np.sqrt(3), -1], [0, 2]]) / 3
_HEX2XY = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, 3 / 2.]])


def assert_hex_coordinate(u, axis=0):
    assert np.shape(u)[axis] == 2


def xy2hex(u):
    return _XY2HEX.dot(u) / 3


def hex2xy(u):
    return _HEX2XY.dot(u)


def neighbors():
    return np.array(((+1, 0), (+1, -1), (0, -1),
                     (-1, 0), (-1, +1), (0, +1)))


def circule_indices(r):
    ij = ((i, j) for i in range(-r, r + 1) for j in range(max(-r, -r - i), min(r, r - i) + 1))
    return np.array(tuple(ij)).T


def square_indices(shape):
    assert len(shape) == 2
    ij = ((i - j // 2, j) for i, j in itertools.product(*map(range, shape)))
    return np.array(tuple(ij)).T


def distance(u, v, axis=0):
    assert_hex_coordinate(u, axis=axis)
    assert_hex_coordinate(v, axis=axis)

    diff = np.subtract(u, v)
    diff_sum = np.sum(diff, axis=axis)

    return np.max((np.max(np.abs(diff), axis=axis), np.abs(diff_sum)), axis=0)
