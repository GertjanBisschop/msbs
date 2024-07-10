import dataclasses
import math
import numpy as np
import random
import msprime
import tskit

from scipy.stats import gamma

from msbs import ancestry
from msbs import utils


@dataclasses.dataclass
class Simulator(ancestry.SuperSimulator):
    U: float = 2e-3
    s: float = 1e-3

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.lineages = []
        self.num_lineages = 0
        self.rate_f = self.instantaneous_rate()

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.num_lineages += 1

    def instantaneous_rate(self):
        R = self.r * self.L

        def _instantaneous_rate(t):
            if t == 0:
                scaling = 0.0
            else:
                a = (1 - np.exp(-self.s * t)) ** 2
                b = (
                    self.s
                    / (self.s + R / 2)
                    * (1 - np.exp(-self.s * t - R * t / 2)) ** 2
                )
                c = utils.generalized_gamma(self.s * t, 2 * self.s * t)
                d = utils.generalized_gamma(R * t / 2, R * t + 2 * self.s * t)
                tot = a - b + 2 * self.s * t * (c - d)
                scaling = -2 * self.U / R * tot

            Ne = self.Ne * np.exp(scaling)
            num_pairs = math.comb(self.num_lineages, 2)
            return num_pairs / (self.ploidy * Ne)

        return _instantaneous_rate

    def instantaneous_rate_const(self):
        def _instantaneous_rate_const(t):
            num_pairs = math.comb(self.num_lineages, 2)
            return num_pairs / (self.ploidy * self.Ne)

        return _instantaneous_rate_const

    def run(self, simplify=True):
        return self._sim(simplify)

    def _sim(self, simplify=True):
        """
        Experimental implementation of coalescent with local Ne map along genome.

        NOTE! This hasn't been statistically tested and is probably not correct.
        """

        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            self.insert_lineage(ancestry.Lineage(len(nodes), segment_chain))
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not self.stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            t_ca = utils.sample_nhpp(self.rate_f, self.rng, t, jump=100)
            t_ca -= t
            t_inc = min(t_re, t_ca)
            t += t_inc
            if t_inc == t_re:  # recombination
                left_lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(
                    left_lineage.left + 1, left_lineage.right
                )
                assert left_lineage.left < breakpoint < left_lineage.right
                right_lineage = left_lineage.split(breakpoint)
                self.insert_lineage(right_lineage)
                child = left_lineage.node
                assert right_lineage.node == child

            else:  # common ancestor event
                a = self.remove_lineage(self.rng.randrange(self.num_lineages))
                b = self.remove_lineage(self.rng.randrange(self.num_lineages))
                c = ancestry.Lineage(len(nodes), [])
                for interval, intersecting_lineages in ancestry.merge_ancestry([a, b]):
                    # if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )

                nodes.append(ancestry.Node(time=t))
                self.insert_lineage(c)

        return self.finalise(tables, nodes, simplify)


dataclasses.dataclass


class StepWiseSimulator(ancestry.SuperSimulator):
    U: float = 2e-3
    s: float = 1e-3

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.lineages = []
        self.num_lineages = 0

    def run(self, demography, seed=None, simplify=True):
        return self._sim(demography, seed, simplify)

    def _sim(self, demography, seed, simplify):
        return msprime.sim_ancestry(
            self.n,
            demography=demography,
            recombination_rate=self.r,
            sequence_length=self.L,
            random_seed=seed,
        )
