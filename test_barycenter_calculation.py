from wasserstein import calc_wasserstein_barycenter, calc_wasserstein_barycenter_old
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def draw_conf2D(mu, cov, ax, edgecolor='k', facecolor='none', n_std=2, **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == '__main__':
    N = 4
    mus = np.array([np.array([0, 1]),
           np.array([0.5, -1]),
           np.array([1.5, .5]),
           np.array([-2, -2])])
    covs = np.array([np.array([[1., -.7], [-.7, 1.]]),
            np.array([[.7, -.4], [-.4, 1.]]),
            np.array([[1., .2], [.2, 1.]]),
            np.array([[2., -.9], [-.9, .5]])])
    mu, cov = calc_wasserstein_barycenter(mus, covs, max_iterations=50)
    mu_old, cov_old = calc_wasserstein_barycenter_old(mus, covs, max_iterations=50)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(N):
        draw_conf2D(mus[i], covs[i], ax)
    draw_conf2D(mu, cov, ax, edgecolor='r')
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    # ax[1].plot(log)