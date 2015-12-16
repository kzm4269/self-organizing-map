import numpy as np


def norm_bmu(ord=None):
    """
    入力ベクトルとの差のノルムが最小となる重みベクトルの番号をBMUとして返す関数を生成する.
    :param ord: @see numpy.linalg.norm
    :return: 重みベクトル列と入力ベクトルからBMUを返す関数
    """

    def _norm_bmu(weights, x):
        return np.argmin(np.linalg.norm(np.subtract(weights, x), ord=ord, axis=1))

    return _norm_bmu


def distance_bmu(d):
    """
    入力ベクトルとの距離が最小となる重みベクトルの番号をBMUとして返す関数を生成する.
    :param d: 距離関数
    :return: 重みベクトル列と入力ベクトルからBMUを返す関数
    """

    def _distance_bmu(weights, x):
        return np.argmin(tuple(d(w, x) for w in weights))

    return _distance_bmu
