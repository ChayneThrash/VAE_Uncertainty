import numpy as np
from scipy.linalg import sqrtm


def calc_wasserstein_barycenter(mus, covariances, max_iterations=1000, eps=1e-10):
    N = mus.shape[0]
    barycenter_mu = np.mean(mus, axis=0)
    barycenter_covariance = np.identity(barycenter_mu.shape[0])
    for i in range(max_iterations):
        covariance_sqrt = sqrtm(barycenter_covariance)
        inv_covariance_sqrt = np.linalg.inv(covariance_sqrt)
        covariance_sqrt = np.expand_dims(covariance_sqrt, axis=0)

        inner_term = covariance_sqrt @ covariances @ covariance_sqrt
        for k in range(N):
            inner_term[k] = sqrtm(inner_term[k])
        inner_term_summed = ((1 / N) * inner_term).sum(axis=0)
        inner_term_summed = inner_term_summed @ inner_term_summed

        new_barycenter_covariance = inv_covariance_sqrt @ inner_term_summed @ inv_covariance_sqrt

        change = np.sum(np.abs(barycenter_covariance - new_barycenter_covariance))
        print("Change at iteration {0}: {1}".format(i, change))
        barycenter_covariance = new_barycenter_covariance

        if change <= eps:
            break
    return barycenter_mu, barycenter_covariance


def calc_wasserstein_barycenter_old(mus, covariances, max_iterations, eps=1e-10):
    N = mus.shape[0]
    mu = np.mean(mus, 0)
    A = np.eye(mus[0].shape[0])
    log = []
    for m in range(max_iterations):
        A_sq = sqrtm(A)
        A_sq = np.expand_dims(A_sq, axis=0)
        inner_term = A_sq @ covariances @ A_sq
        for k in range(N):
            inner_term[k] = sqrtm(inner_term[k])
        A_new = np.sum((1 / N) * (inner_term), axis=0)

        change = np.sum(np.abs(A - A_new))
        print("Change at iteration {0}: {1}".format(m, change))
        A = A_new
        if change < eps:
            break
    return mu, A