import math
import msprime
import numpy as np
import random
import scipy.stats

from scipy.special import gammaln, exp1
from typing import Callable, Tuple, Iterable, Generator


def combinadic_map(sorted_pair: Iterable) -> Tuple:
    """
    Maps a pair of indices to a unique integer.
    """
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx: int, k=2) -> Generator[int, int, int]:
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


def zero_sup_rate(
    rate_f: Callable, rng: random.Random, upper_t: float, jump: float = 0.1
) -> Tuple:
    """
    This is a slow solution to the issue of sometimes having a very low event rate.
    """
    sup_rate = rate_f(upper_t)
    i = 0
    while sup_rate == 0 and i < 10:
        jump *= 10
        upper_t += jump
        sup_rate = rate_f(upper_t)
        i += 1

    return sup_rate, upper_t


def sample_nhpp(rate_f: Callable, rng: random.Random, start_time=0, jump=0.1) -> float:
    """
    Algorithm to draw the first interevent time for a
    non-homogeneous poisson process starting at start_time.
    Algorithm adapted from Introduction to
    Probability Models by Sheldon Ross (11th edition, p 673).
    """
    upper_t_interval = jump + start_time
    sup_rate = rate_f(upper_t_interval)
    if sup_rate == 0:
        sup_rate, upper_t_interval = zero_sup_rate(rate_f, rng, upper_t_interval, jump)
        if sup_rate == 0:
            return math.inf
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
            if sup_rate == 0:
                sup_rate, upper_t_interval = zero_sup_rate(
                    rate_f, rng, upper_t_interval, jump
                )
                if sup_rate == 0:
                    return math.inf
            w = adjust_w * old_sup_rate / sup_rate

    return new_time


def markov_chain_expected_absorption(b_inv):
    result = np.zeros(b_inv.shape[0])
    e = np.ones_like(result)
    z = np.zeros_like(result)
    for i in range(b_inv.shape[0]):
        z[i] = 1
        result[i] = z @ -b_inv @ e
        z[i] = 0

    return np.flip(result)


def generalized_gamma(x, y):
    return exp1(x) - exp1(y)


def stepwise_factory(initial_size, time_steps, ne_curve):
    demography = msprime.Demography()
    demography.add_population(initial_size=initial_size)
    for i in range(time_steps.size):
        demography.add_population_parameters_change(
            time_steps[i], initial_size=ne_curve[i]
        )

    return demography
