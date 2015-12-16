import numpy as np


def gaussian(pdists, alpha, beta):
    """
    ガウス関数型の近傍関数を返す.
    :param pdists: ニューロン間距離行列
    :param alpha: 時刻 t におけるガウス関数の振幅を返す関数
    :param beta: 時刻 t におけるガウス関数の分散を返す関数
    :return: 時刻 t における各ニューロン間の近傍関数の値を返す関数
    """

    def neighborhood(t):
        return alpha(t) * np.exp(-pdists ** 2 / (2 * beta(t)))

    return neighborhood