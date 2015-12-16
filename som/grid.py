import numpy as np
from scipy.spatial import distance
from . import hexgrid


def grid_pairs(pdists, weights=None):
    """
    隣接した2つニューロンまたはそれらの重みベクトルの組を返す.
    :param pdists: ニューロン間距離行列
    :param weights: 重みベクトル列
    :return: 隣接したニューロンの番号の組の列. 重みベクトルを指定した場合は重みベクトルの組の列.
    """
    if weights is None:
        where = np.where(pdists <= np.min(pdists[pdists > 0]))
        return ((i, j) for i, j in zip(*where) if i < j)
    return (weights[pair] for pair in map(list, grid_pairs(pdists)))


def square_grid(shape):
    """
    正方格子型SOMのニューロン間距離行列と初期重みベクトル列を返す.
    :param shape: SOMの形状
    :return: ニューロン間距離行列と初期重みベクトル列
    """
    indices = np.indices(shape).reshape((len(shape), np.prod(shape)))
    pdists = distance.squareform(distance.pdist(X=indices.T, metric=distance.cityblock))
    weights = indices.T
    return pdists, weights


def hex_grid(shape):
    """
    六方格子型SOMのニューロン間距離行列と初期重みベクトル列を返す.
    :param shape: SOMの形状
    :return: ニューロン間距離行列と初期重みベクトル列
    """
    indices = hexgrid.square_indices(shape).reshape((len(shape), np.prod(shape)))
    pdists = distance.squareform(distance.pdist(X=indices.T, metric=hexgrid.distance))
    weights = hexgrid.hex2xy(indices).T
    return pdists, weights
