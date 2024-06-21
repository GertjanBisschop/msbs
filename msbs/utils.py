import math
import numpy as np
import random
import scipy.stats

from scipy.special import gammaln
from typing import Callable, Tuple, Iterable


def combinadic_map(sorted_pair: Iterable) -> Tuple:
    """
    Maps a pair of indices to a unique integer.
    """
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx: int, k=2) -> int:
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


def pairwise_products(v: np.ndarray) -> np.ndarray:
    assert len(v.shape) == 1
    n = v.shape[0]
    m = v.reshape(n, 1) @ v.reshape(1, n)

    return m[np.tril_indices_from(m, k=-1)].ravel()


def poisson_pmf(x: float, mu: float) -> float:
    out = -mu + x * np.log(mu) - gammaln(x + 1)
    return np.exp(out)


def poisson_cmf(x: float, mu: float) -> float:
    return scipy.stats.poisson.cdf(x, mu)


def sample_nhpp(rate_f: Callable, rng: random.Random, start_time=0, jump=0.1) -> float:
    """
    Algorithm to draw the first interevent time for a
    non-homogeneous poisson process starting at start_time.
    Algorithm adapted from Introduction to
    Probability Models by Sheldon Ross (11th edition, p 673).
    """
    upper_t_interval = jump + start_time
    sup_rate = rate_f(upper_t_interval)
    new_time = start_time
    w = rng.expovariate(sup_rate)

    while True:
        if new_time + w < upper_t_interval:
            new_time += w
            u = rng.uniform(0, 1)
            if u < rate_f(new_time) / sup_rate:
                break
            w = rng.expovariate(sup_rate)
        else:
            adjust_w = w - upper_t_interval + new_time
            new_time = upper_t_interval
            upper_t_interval += jump
            old_sup_rate = sup_rate
            sup_rate = rate_f(upper_t_interval)
            w = adjust_w * old_sup_rate / sup_rate

    return new_time
