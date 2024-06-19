import math
import numpy as np

from scipy.special import gammaln


def combinadic_map(sorted_pair):
    """
    Maps a pair of indices to a unique integer.
    """
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx, k=2):
    """
    Maps a unique index to a unique pair.
    """
    while k > 0:
        i = k - 1
        num_combos = 0
        while num_combos <= idx:
            i += 1
            num_combos = math.comb(i, k)
        yield i - 1
        idx -= math.comb(i - 1, k)
        k -= 1


def pairwise_products(v: np.ndarray):
    assert len(v.shape) == 1
    n = v.shape[0]
    m = v.reshape(n, 1) @ v.reshape(1, n)

    return m[np.tril_indices_from(m, k=-1)].ravel()


def poisson_pmf(x, mu):
    out = -mu + x * np.log(mu) - gammaln(x + 1)
    return np.exp(out)
