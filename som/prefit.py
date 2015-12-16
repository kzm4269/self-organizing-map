import numpy as np
from sklearn.decomposition.pca import PCA


def pca_prefit(weights, xs):
    """
    SOMの初期値を計算するための前処理.
    線形変換によって重みベクトル列の主成分とその固有値を入力ベクトル列のものと一致させる.
    :param weights: 初期重みベクトル列
    :param xs: 入力ベクトル列
    :return: 前処理した重みベクトル列
    """
    n = np.shape(xs)[1]
    pca_w = PCA(n_components=n)
    pca_w.fit(weights)
    pca_x = PCA(n_components=n)
    pca_x.fit(xs)

    mean_w = np.mean(weights, axis=0)
    mean_x = np.mean(xs, axis=0)
    com_w = pca_w.components_
    com_x = pca_x.components_
    var_w = pca_w.explained_variance_
    var_x = pca_x.explained_variance_

    var_w[var_w == 0] = np.max(var_w) * 1e-6
    new_w = (weights - mean_w).dot(com_w.T) / np.sqrt(var_w)
    new_w = (new_w * np.sqrt(var_x)).dot(com_x) + mean_x

    return new_w
