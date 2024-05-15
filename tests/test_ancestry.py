import pytest
import msbs.ancestry as ancestry

class TestSimBasic:
    def test_simple(self):
        L = 100
        rho = 0.1
        n = 4
        Ne = 1
        seed = 12

        sim = ancestry.Simulator(L, rho, n, Ne, seed)
        ts = sim.sim_coalescent()
        assert ts.num_trees > 1
