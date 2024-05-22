import collections
import random
import math
import dataclasses
import numpy as np
import tskit

from typing import List
from typing import Any

from msbs import ancestry


@dataclasses.dataclass
class BinLineage(ancestry.Lineage):
    def __post_init__(self):
        self.bins = None

    def set_value(self):
        self.value = np.sum(self.bins)

    def set_bins(self, mean_muts, num_bins, rng):
        self.bins = rng.poisson(mean_muts, num_bins)

    def split(self, breakpoint, mean_muts, binwidth, rng):

        left_ancestry = []
        right_ancestry = []

        for interval in self.ancestry:
            if interval.right <= breakpoint:
                left_ancestry.append(interval)
            elif interval.left >= breakpoint:
                right_ancestry.append(interval)
            else:
                assert interval.left < breakpoint < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=breakpoint))
                right_ancestry.append(dataclasses.replace(interval, left=breakpoint))
        self.ancestry = left_ancestry
        right_lin = BinLineage(self.node, right_ancestry)
        right_lin.bins = self.bins.copy()

        # modify bin counts left and right of breakpoint
        # issues here with correct boundary!!
        i = breakpoint // binwidth  # bin at breakpoint index
        value = self.bins[i]
        p = breakpoint % binwidth / binwidth
        q = -breakpoint % binwidth / binwidth
        value_left = rng.binomial(n=value, p=p)
        self.bins[i] = value_left + rng.binomial(n=mean_muts, p=q)
        right_lin.bins[i] = value - value_left + rng.binomial(n=mean_muts, p=p)

        return right_lin


@dataclasses.dataclass
class BinSimulator(ancestry.SuperSimulator):
    U: float = 5e-2  # number of mutations per generation per ind
    s: float = 0.01
    num_bins: int = 10

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.lineages = []
        self.num_lineages = 0
        self.haplo_dict = collections.defaultdict(list)
        self.bins = np.linspace(0, self.L, num=self.num_bins, endpoint=False)

    def remove_lineage(self, lineage_id):
        lin = self.lineages.pop(lineage_id)
        self.num_lineages -= 1
        self.haplo_dict[lin.value].remove(lin)

        return lin

    def insert_lineage(self, lineage):
        self.lineages.append(lineage)
        self.haplo_dict[lineage.value].append(lineage)
        self.num_lineages += 1

    def count_i_types(self):
        return {i: len(self.haplo_dict[i]) for i in self.haplo_dict.keys()}

    def mutation_event(self, total_mass):
        for lineage in self.lineages:
            if self.rng.random() > lineage.value / total_mass:
                break

        nonzero = np.nonzero(lineage.bins)[0]
        assert nonzero.size > 0
        binindex = nonzero[self.rng.randrange(nonzero.size)]
        self.haplo_dict[lineage.value].remove(lineage)
        lineage.bins[binindex] -= 1
        lineage.value -= 1
        self.haplo_dict[lineage.value].append(lineage)

    def common_ancestor_waiting_time_from_rate(self, rate):
        u = self.rng.expovariate(rate)
        return u

    def common_ancestor_waiting_time(self):
        # perform all pairwise weighted contributions
        n = self.num_lineages
        rate = 1 / n
        return self.common_ancestor_waiting_time_from_rate(rate)

    def run(self, simplify=True):
        return self._sim_bins(simplify)

    def _sim_bins(self, simplify):
        rng = np.random.default_rng(self.rng.randint(1, 2**16))
        tables = tskit.TableCollection(self.L)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        nodes = []
        mean_load = self.U / self.s
        for _ in range(self.n * self.ploidy):
            segment_chain = [ancestry.AncestryInterval(0, self.L, 1)]
            lineage = BinLineage(len(nodes), segment_chain)
            lineage.set_bins(mean_load, self.num_bins, rng)
            self.insert_lineage(lineage)
            nodes.append(ancestry.Node(time=0, flags=tskit.NODE_IS_SAMPLE))

        t = 0
        while not super().stop_condition():
            lineage_links = [
                lineage.num_recombination_links for lineage in self.lineages
            ]
            total_links = sum(lineage_links)
            re_rate = total_links * self.r
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            t_ca = self.common_ancestor_waiting_time()
            num_mutations = sum(lin.value for lin in self.lineages)
            t_mut = self.rng.expovariate(self.s * num_mutations)
            t_inc = min(t_re, t_ca, t_mut)
            t += t_inc

            if t_inc == t_re:  # recombination
                pass
            elif t_inc == t_ca:  # common ancestor event
                pass
            else:  # mutation
                self.mutation_event(num_mutations)
            break

        return self.finalise(tables, nodes, simplify)
