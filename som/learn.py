import numpy as np

def batch_learn(weights, xs, bmu, neighborhood):
    """
    バッチ型SOMの学習則.
    :param weights: 現在の重みベクトル列
    :param xs: 入力ベクトル列
    :param bmu: 重みベクトル列と入力ベクトルに対してBMUを返す関数
    :param neighborhood: 近傍行列. 各ニューロンから各ニューロンへの近さ.
    :return: 更新された重みベクトル列
    """
    num = np.zeros_like(weights)
    den = np.zeros((len(weights), 1))
    for x in xs:
        c = bmu(weights, x)
        h = np.reshape(neighborhood[c], den.shape)
        num += np.subtract(x, weights) * h
        den += h
    return weights + num / (den + np.finfo(np.float).eps)


def batch_fit(weights, xs, bmu, neighborhood, n=1, callback=None):
    """
    SOMの学習を行う.
    :param weights: 初期重みベクトル列
    :param xs: 入力ベクトル列
    :param bmu: 重みベクトル列と入力ベクトルに対してBMUを返す関数
    :param neighborhood: 時刻 t における各ニューロン間の近傍関数の値を返す関数
    :param n: 反復回数
    :param callback: コールバック関数. 反復1回ごとに呼ばれる.
    :return: 学習後の重みベクトル
    """
    assert n > 0
    for t in np.arange(0, 1, 1. / n):
        weights = batch_learn(weights, xs, bmu, neighborhood(t))
        if callback is not None:
            callback(t, weights)
    return weights