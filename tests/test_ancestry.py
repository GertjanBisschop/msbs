import pytest
import numpy as np
import msbs.ancestry as ancestry
import msbs.bins as bins
import msbs.zeng as zeng

import time


class TestSimAncestry:
    def test_simple(self):
        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000
        seed = 12
        b_map = ancestry.BMap(
            np.array([0, L // 2, 3 * L // 4, L]), np.array([1.0, 0.01, 1.0])
        )

        sim = ancestry.Simulator(L, r, n, Ne, seed=seed, B=b_map)
        ts = sim.run()
        assert ts.num_trees > 1
        # ts.dump('bs.trees')


class TestSimBins:
    def test_simple(self):
        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000
        seed = 12

        sim = bins.BinSimulator(L, r, n, Ne, seed=seed, num_bins=5)
        ts = sim.run()
        assert False

    def test_recombination(self):
        rng = np.random.default_rng(10)
        alist = [ancestry.AncestryInterval(19, 30, 0)]
        left = bins.BinLineage(0, alist, 10)
        left.set_bins(10, 10, rng)
        pre_split = left.bins.copy()
        right = left.split(20, 10, 10, rng)
        assert np.all(np.hstack([left.bins[:2], right.bins[2:]]) == pre_split)


class TestSimZeng:
    def test_recombination(self):
        rng = np.random.default_rng(10)
        alist = [ancestry.AncestryInterval(19, 30, 0)]
        mean_load = 10
        mutations = np.array([10, 1, 12, 9], dtype=np.int64)
        breakpoints = np.array([0, 10, 20, 30, 100])

        # split [0, 10, 20, 23, 100] [a, b, c, d]
        # and [0, 23, 30, 100] [x, y]

        left = zeng.ZLineage(0, alist, np.sum(mutations), mutations, breakpoints)
        left.set_fitness()

        right = left.split(23, mean_load, rng)
        assert np.array_equal(left.breakpoints, np.array([0, 10, 20, 23, 100]))
        assert np.array_equal(right.breakpoints, np.array([0, 23, 30, 100]))

    def test_recombination_b(self):
        rng = np.random.default_rng(10)
        alist = [ancestry.AncestryInterval(0, 100, 0)]
        mean_load = 10
        mutations = np.array([22], dtype=np.int64)
        breakpoints = np.array([0, 100])

        left = zeng.ZLineage(0, alist, np.sum(mutations), mutations, breakpoints)
        left.set_fitness()

        right = left.split(48, mean_load, rng)
        assert np.array_equal(left.breakpoints, np.array([0, 48, 100]))
        assert left.mutations[0] + right.mutations[1] == 22
        assert np.array_equal(right.breakpoints, np.array([0, 48, 100]))

    def test_coalescing(self):
        a = zeng.ZLineage(
            0,
            [],
            1,
            np.array([1, 0]),
            np.array([0, 0.5, 1]),
        )
        b = zeng.ZLineage(
            1,
            [],
            1,
            np.array([0, 1]),
            np.array([0, 0.25, 1]),
        )
        bool_coal, c = a.coalescing(b)
        assert bool_coal
        assert np.array_equal(c.breakpoints, np.array([0, 0.25, 0.5, 1.0]))
        assert np.array_equal(c.mutations, np.array([0, 1, 0]))

    def test_coal_no_muts(self):
        a = zeng.ZLineage(
            0,
            [],
            1,
            np.array([0]),
            np.array([0, 1]),
        )
        b = zeng.ZLineage(
            1,
            [],
            1,
            np.array([0]),
            np.array([0, 1]),
        )
        bool_coal, c = a.coalescing(b)
        assert bool_coal

        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000
        U = 0.0
        sim = zeng.ZSimulator(L, r, n, Ne, U=U)
        sim.insert_lineage(a)
        print(a.prob_i(), b.prob_i())
        sim.insert_lineage(b)
        print(sim.get_coal_rate(U))
        assert False

    @pytest.mark.parametrize("seed", [12, 89901, 1303])
    def test_simple(self, seed):
        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000

        sim = zeng.ZSimulator(L, r, n, Ne, seed=seed)
        ts = sim.run()

        print(
            f"ca: {sim.num_ca_events}, re: {sim.num_re_events}, mu: {sim.num_mu_events}"
        )
        assert False
        for tree in ts.trees():
            assert tree.num_roots == 1

    def test_zero(self):
        seed = 4684
        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000

        sim = zeng.ZSimulator(L, r, n, Ne, seed=seed, U=0.0)
        ts = sim.run()

        print(
            f"ca: {sim.num_ca_events}, re: {sim.num_re_events}, mu: {sim.num_mu_events}"
        )
        assert False
        for tree in ts.trees():
            assert tree.num_roots == 1

    def test_range(self):
        s = 0.01
        U_range = np.linspace(0.001, 0.5, num=5)

        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000

        for u in U_range:
            sim = zeng.ZSimulator(L, r, n, Ne, U=u, seed=99)
            start = time.time()
            ts = sim.run()
            print(time.time() - start)
            print(sim.num_ca_events, sim.num_re_events, sim.num_mu_events)
            print(sim.U / sim.s, sim.U * (1 - sim.s) / sim.s)
            print("-----------------")
        assert False
