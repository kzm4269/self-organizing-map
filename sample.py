import numpy as np
import matplotlib.pyplot as plt
import som


def sample_data(n, e=None, v=None, components=None, ndim=2):
    """多変量正規分布"""
    e = np.zeros(ndim) if e is None else e
    v = np.ones(ndim) if v is None else v
    components = np.eye(ndim) if components is None else components
    return np.random.randn(n, ndim).dot(np.diag(v)).dot(components) + e


def test_fit(xs, callback):
    # grid
    pdists, weights = som.grid.hex_grid((4, 3))

    # prefit
    weights = som.prefit.pca_prefit(np.hstack((weights, np.zeros((len(pdists), 1)))), xs)

    # fit
    weights = som.batch_fit(weights=weights,
                            xs=xs,
                            bmu=som.bmu.norm_bmu(),
                            neighborhood=som.neighborhood.gaussian(pdists,
                                                                   alpha=lambda t: 1.0 * (1 - t),
                                                                   beta=lambda t: 0.3 * (1 - t)),
                            n=20,
                            callback=lambda t, w: callback(t, pdists, w))

    # finish
    callback(1, pdists, weights)


def main(plot=False, proj_dim=2):
    # data set
    n = 100
    xs = np.ndarray((0, proj_dim))
    for i in np.random.random(9) + 1:
        scale = (0.1 + np.random.random(proj_dim) / 0.9) * 0.1
        center = np.random.random(proj_dim)
        rot = np.dot(*np.linalg.svd(np.random.random((proj_dim, proj_dim)))[::2])
        xs = np.vstack([xs, sample_data(n * i, e=center, v=scale, components=rot, ndim=proj_dim)])

    # plot config
    ax = plt.figure().add_subplot(111, projection='3d' if proj_dim == 3 else None)
    ax.figure.canvas.window().setGeometry(0, 30, 800, 800)
    ax.set_axis_bgcolor('k')

    def fit_callback(t, pdists, weights):
        print('\rfitting {:3.0f}%'.format(100 * t), end='')
        if not plot:
            return
        ax.clear()
        ax.plot(*xs.T, linestyle='', marker='.', color='w', alpha=0.5)
        for pair in som.grid.grid_pairs(pdists, weights):
            plt.plot(*pair.T, linestyle='-', marker='o', color='r', alpha=0.75)

        center = np.mean(xs, axis=0)
        lim = np.array((np.min(xs - center), np.max(xs - center)))
        ax.set_xlim(lim + center[0])
        ax.set_ylim(lim + center[1])
        if proj_dim == 3:
            ax.set_zlim(lim + center[2])
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        plt.show(block=False)

    # fit
    test_fit(xs, fit_callback)

    if plot:
        ax.set_aspect('auto')
        plt.show()


def profit_main():
    import cProfile
    import pstats

    fname = 'pstats'
    cProfile.run('main(plot=False)', fname)
    stats = pstats.Stats(fname)
    stats.sort_stats('cumtime')
    stats.reverse_order()
    stats.print_stats()

    import subprocess
    subprocess.call('gprof2dot -f pstats pstats | dot -Tpng -o gprof.png', shell=True)


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # profit_main()
    main(plot=True)
