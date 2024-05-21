import pytest
import numpy as np
import msbs.ancestry as ancestry

class TestSimBasic:
    def test_simple(self):
        L = 100
        r = 0.1e-4
        n = 4
        Ne = 10_000
        seed = 12
        b_map = ancestry.BMap(np.array([0, L//2, 3*L//4, L]), np.array([1.0, 0.01, 1.0]))
        
        sim = ancestry.Simulator(L, r, n, Ne, seed=seed, B=b_map)
        ts = sim.run()
        assert ts.num_trees > 1
        #ts.dump('bs.trees')
